"""Post-hoc calibration for property predictors.

Fits calibration models (isotonic regression or temperature scaling)
on a held-out calibration set to correct miscalibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationSignal:
    """Result from calibration correction."""

    raw_probability: float
    calibrated_probability: float
    calibration_shift: float  # how much calibration changed the prediction
    confidence: float  # higher if calibration is well-supported


class Calibrator:
    """Post-hoc probability calibration for property predictors.

    Fits on held-out calibration data where we know ground truth labels
    and raw model predictions. Then corrects new predictions.
    """

    def __init__(self, method: str = "isotonic") -> None:
        self.method = method
        self._fitted = False

        if method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip")
        elif method == "temperature_scaling":
            self._temperature = 1.0
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def fit(self, raw_probs: np.ndarray, true_labels: np.ndarray) -> None:
        """Fit calibration on held-out data.

        Args:
            raw_probs: (n_samples,) array of raw model probabilities.
            true_labels: (n_samples,) array of binary ground truth labels.
        """
        if self.method == "isotonic":
            self._model.fit(raw_probs, true_labels)
        elif self.method == "temperature_scaling":
            self._temperature = self._fit_temperature(raw_probs, true_labels)

        self._fitted = True
        self._n_calibration_samples = len(raw_probs)

    def calibrate(self, raw_prob: float) -> CalibrationSignal:
        """Calibrate a single raw probability.

        Args:
            raw_prob: Raw model probability (0-1).

        Returns:
            CalibrationSignal with corrected probability.
        """
        if not self._fitted:
            # No calibration data: return raw with low confidence
            return CalibrationSignal(
                raw_probability=raw_prob,
                calibrated_probability=raw_prob,
                calibration_shift=0.0,
                confidence=0.3,
            )

        if self.method == "isotonic":
            calibrated = float(self._model.predict([raw_prob])[0])
        elif self.method == "temperature_scaling":
            logit = np.log(raw_prob / (1 - raw_prob + 1e-10) + 1e-10)
            scaled_logit = logit / self._temperature
            calibrated = float(1.0 / (1.0 + np.exp(-scaled_logit)))

        shift = abs(calibrated - raw_prob)

        # Confidence in calibration: higher with more calibration data,
        # lower when calibration makes large corrections (suggests model is poorly calibrated)
        data_confidence = min(1.0, self._n_calibration_samples / 500)
        shift_penalty = max(0.0, 1.0 - 2.0 * shift)
        confidence = data_confidence * shift_penalty

        return CalibrationSignal(
            raw_probability=round(raw_prob, 4),
            calibrated_probability=round(calibrated, 4),
            calibration_shift=round(shift, 4),
            confidence=round(confidence, 4),
        )

    @staticmethod
    def _fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
        """Optimize temperature via grid search on NLL."""
        from scipy.optimize import minimize_scalar

        def nll(T: float) -> float:
            logits = np.log(probs / (1 - probs + 1e-10) + 1e-10)
            scaled = 1.0 / (1.0 + np.exp(-logits / T))
            scaled = np.clip(scaled, 1e-10, 1 - 1e-10)
            return -float(np.mean(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled)))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        return float(result.x)
