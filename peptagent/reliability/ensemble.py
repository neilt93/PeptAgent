"""Ensemble disagreement for reliability estimation.

When multiple models predict the same property, their disagreement
signals epistemic uncertainty. High disagreement → low confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnsembleSignal:
    """Result from ensemble disagreement analysis."""

    mean_prediction: float
    std_prediction: float
    confidence: float  # 0-1, derived from std
    n_models: int


class EnsembleEstimator:
    """Estimate reliability from ensemble disagreement.

    Given predictions from multiple models for the same property,
    computes disagreement-based confidence. Used per-property.
    """

    def __init__(self, disagreement_threshold: float = 0.3) -> None:
        self.disagreement_threshold = disagreement_threshold

    def estimate(self, predictions: list[float]) -> EnsembleSignal:
        """Compute ensemble signal from multiple model predictions.

        Args:
            predictions: List of probability predictions from ensemble members.

        Returns:
            EnsembleSignal with confidence derived from agreement level.
        """
        if len(predictions) < 2:
            return EnsembleSignal(
                mean_prediction=predictions[0] if predictions else 0.0,
                std_prediction=0.0,
                confidence=0.5,  # unknown confidence with single model
                n_models=len(predictions),
            )

        mean = float(np.mean(predictions))
        std = float(np.std(predictions))

        # Map std to confidence: 0 std → 1.0 confidence, threshold std → 0.0
        confidence = max(0.0, 1.0 - std / self.disagreement_threshold)

        return EnsembleSignal(
            mean_prediction=mean,
            std_prediction=std,
            confidence=confidence,
            n_models=len(predictions),
        )
