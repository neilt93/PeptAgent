"""Aggregate reliability signals into a unified report for the agent.

This is the core of the reliability layer — it combines ensemble disagreement,
OOD detection, and calibration signals into a single ReliabilityReport that
gets serialized into the LLM context for reliability-aware decision making.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from peptagent.reliability.calibration import Calibrator
from peptagent.reliability.ensemble import EnsembleEstimator
from peptagent.reliability.ood import OODDetector


@dataclass
class ReliabilityReport:
    """Unified reliability assessment for a peptide candidate.

    Serialized into the LLM context so the agent can reason about
    prediction reliability when making design decisions.
    """

    overall_confidence: float
    per_property_confidence: dict[str, float]
    ood_score: float | None
    is_ood: bool
    flags: list[str]
    recommendation: str

    def to_context_string(self) -> str:
        """Format for inclusion in LLM conversation context."""
        lines = [
            f"RELIABILITY REPORT (overall confidence: {self.overall_confidence:.2f})",
        ]
        if self.flags:
            lines.append(f"  FLAGS: {'; '.join(self.flags)}")
        for prop, conf in self.per_property_confidence.items():
            level = "HIGH" if conf > 0.7 else "MEDIUM" if conf > 0.4 else "LOW"
            lines.append(f"  {prop}: confidence={conf:.2f} ({level})")
        if self.ood_score is not None:
            lines.append(f"  OOD distance: {self.ood_score:.4f} ({'OUT-OF-DISTRIBUTION' if self.is_ood else 'in-distribution'})")
        lines.append(f"  RECOMMENDATION: {self.recommendation}")
        return "\n".join(lines)


class ReliabilityAggregator:
    """Combine ensemble, OOD, and calibration signals.

    The aggregator is the bridge between the reliability layer and the
    agent loop. It takes raw tool results, computes per-signal confidence,
    and produces a ReliabilityReport the agent can act on.
    """

    def __init__(
        self,
        ensemble: EnsembleEstimator | None = None,
        ood: OODDetector | None = None,
        calibrators: dict[str, Calibrator] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.ensemble = ensemble or EnsembleEstimator()
        self.ood = ood
        self.calibrators = calibrators or {}
        # Relative importance of each signal type
        self.weights = weights or {
            "ensemble": 0.4,
            "ood": 0.3,
            "calibration": 0.3,
        }

    def score(
        self,
        sequence: str,
        tool_results: dict,
        embedding: np.ndarray | None = None,
    ) -> ReliabilityReport:
        """Produce a unified reliability report for a peptide candidate.

        Args:
            sequence: The peptide sequence.
            tool_results: Dict mapping property name → ToolResult from predictors.
            embedding: ESM-2 embedding for OOD detection (optional).

        Returns:
            ReliabilityReport with overall and per-property confidence.
        """
        flags = []
        per_property_confidence = {}
        all_confidences = []

        # --- Per-property analysis ---
        for prop_name, result in tool_results.items():
            prop_confidences = []

            # Ensemble signal (from tool result metadata)
            if result.metadata.get("individual_probs"):
                ensemble_signal = self.ensemble.estimate(result.metadata["individual_probs"])
                prop_confidences.append(
                    ("ensemble", ensemble_signal.confidence, self.weights["ensemble"])
                )
                if ensemble_signal.confidence < 0.4:
                    flags.append(f"High ensemble disagreement on {prop_name}")

            # Calibration signal
            if prop_name in self.calibrators and result.value.get("probability") is not None:
                cal_signal = self.calibrators[prop_name].calibrate(result.value["probability"])
                prop_confidences.append(
                    ("calibration", cal_signal.confidence, self.weights["calibration"])
                )
                if cal_signal.calibration_shift > 0.2:
                    flags.append(f"Large calibration correction on {prop_name}")

            # Tool's own confidence
            if result.confidence is not None:
                prop_confidences.append(("tool", result.confidence, 0.2))

            if prop_confidences:
                # Weighted average of available signals
                total_weight = sum(w for _, _, w in prop_confidences)
                weighted_conf = sum(c * w for _, c, w in prop_confidences) / total_weight
                per_property_confidence[prop_name] = round(weighted_conf, 4)
                all_confidences.append(weighted_conf)

        # --- OOD analysis (sequence-level, not per-property) ---
        ood_score = None
        is_ood = False
        if self.ood is not None and embedding is not None:
            try:
                ood_signal = self.ood.score(embedding)
                ood_score = ood_signal.distance
                is_ood = ood_signal.is_ood
                all_confidences.append(ood_signal.confidence)
                if is_ood:
                    flags.append(
                        f"Sequence is out-of-distribution (distance={ood_score:.4f}, "
                        f"percentile={ood_signal.percentile:.1f}%)"
                    )
            except RuntimeError:
                flags.append("OOD detector not fitted, skipping OOD check")

        # --- Overall confidence ---
        if all_confidences:
            overall_confidence = round(float(np.mean(all_confidences)), 4)
        else:
            overall_confidence = 0.5
            flags.append("No reliability signals available")

        # --- Generate recommendation ---
        recommendation = self._generate_recommendation(
            overall_confidence, is_ood, flags, per_property_confidence
        )

        return ReliabilityReport(
            overall_confidence=overall_confidence,
            per_property_confidence=per_property_confidence,
            ood_score=ood_score,
            is_ood=is_ood,
            flags=flags,
            recommendation=recommendation,
        )

    @staticmethod
    def _generate_recommendation(
        confidence: float,
        is_ood: bool,
        flags: list[str],
        per_property: dict[str, float],
    ) -> str:
        """Generate a natural-language recommendation for the agent."""
        if confidence > 0.8 and not is_ood:
            return "High confidence — predictions are reliable, proceed with this candidate."

        if is_ood:
            return (
                "Candidate is out-of-distribution. Property predictions may be unreliable. "
                "Consider: (1) modifying toward known peptide space, (2) requesting MD simulation "
                "for validation, or (3) generating alternative candidates."
            )

        low_props = [p for p, c in per_property.items() if c < 0.4]
        if low_props:
            return (
                f"Low confidence on: {', '.join(low_props)}. "
                f"Consider generating variants that are more conservative on these properties, "
                f"or request additional validation for these specific predictions."
            )

        if confidence > 0.5:
            return "Moderate confidence — predictions are usable but treat with caution."

        return (
            "Low overall confidence. Multiple reliability flags raised. "
            "Recommend generating new candidates rather than refining this one."
        )
