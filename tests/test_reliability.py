"""Tests for reliability estimation components."""

import numpy as np
import pytest

from peptagent.reliability.calibration import Calibrator
from peptagent.reliability.ensemble import EnsembleEstimator
from peptagent.reliability.ood import OODDetector


class TestEnsembleEstimator:
    def test_high_agreement(self):
        estimator = EnsembleEstimator(disagreement_threshold=0.3)
        signal = estimator.estimate([0.8, 0.82, 0.79])
        assert signal.confidence > 0.8
        assert signal.std_prediction < 0.05

    def test_high_disagreement(self):
        estimator = EnsembleEstimator(disagreement_threshold=0.3)
        signal = estimator.estimate([0.2, 0.8, 0.5])
        assert signal.confidence < 0.5

    def test_single_model(self):
        estimator = EnsembleEstimator()
        signal = estimator.estimate([0.7])
        assert signal.confidence == 0.5  # unknown with single model
        assert signal.n_models == 1


class TestOODDetector:
    def test_in_distribution(self):
        rng = np.random.RandomState(42)
        train = rng.randn(100, 10)
        detector = OODDetector(method="mahalanobis", threshold_percentile=95)
        detector.fit(train)

        signal = detector.score(np.zeros(10))
        assert not signal.is_ood
        assert signal.confidence > 0.5

    def test_out_of_distribution(self):
        rng = np.random.RandomState(42)
        train = rng.randn(100, 10)
        detector = OODDetector(method="mahalanobis", threshold_percentile=95)
        detector.fit(train)

        signal = detector.score(np.ones(10) * 100)
        assert signal.is_ood
        assert signal.confidence < 0.3

    def test_knn_method(self):
        rng = np.random.RandomState(42)
        train = rng.randn(100, 10)
        detector = OODDetector(method="knn", k_neighbors=5, threshold_percentile=95)
        detector.fit(train)

        signal = detector.score(np.zeros(10))
        assert isinstance(signal.distance, float)

    def test_not_fitted_raises(self):
        detector = OODDetector()
        with pytest.raises(RuntimeError):
            detector.score(np.zeros(10))


class TestCalibrator:
    def test_isotonic(self):
        rng = np.random.RandomState(42)
        raw_probs = rng.rand(200)
        labels = (raw_probs + rng.randn(200) * 0.2 > 0.5).astype(float)

        cal = Calibrator(method="isotonic")
        cal.fit(raw_probs, labels)

        signal = cal.calibrate(0.7)
        assert 0 <= signal.calibrated_probability <= 1
        assert signal.confidence > 0

    def test_temperature_scaling(self):
        rng = np.random.RandomState(42)
        raw_probs = rng.rand(200)
        labels = (raw_probs > 0.5).astype(float)

        cal = Calibrator(method="temperature_scaling")
        cal.fit(raw_probs, labels)

        signal = cal.calibrate(0.8)
        assert 0 <= signal.calibrated_probability <= 1

    def test_unfitted_returns_raw(self):
        cal = Calibrator(method="isotonic")
        signal = cal.calibrate(0.7)
        assert signal.calibrated_probability == 0.7
        assert signal.confidence == 0.3  # low confidence when unfitted
