"""Out-of-distribution detection via embedding distance.

Peptides far from the training distribution in ESM-2 embedding space
are flagged as potentially unreliable — the property predictors have
not seen similar sequences during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class OODSignal:
    """Result from out-of-distribution detection."""

    distance: float
    percentile: float  # where this distance falls in the training distribution
    is_ood: bool
    confidence: float  # 0-1, lower if OOD


class OODDetector:
    """Detect out-of-distribution peptides using embedding distance.

    Fits on training set embeddings, then scores new peptides by their
    distance to the nearest training examples.
    """

    def __init__(
        self,
        method: str = "mahalanobis",
        k_neighbors: int = 10,
        threshold_percentile: float = 95.0,
    ) -> None:
        self.method = method
        self.k_neighbors = k_neighbors
        self.threshold_percentile = threshold_percentile
        self._fitted = False
        self._training_distances: np.ndarray | None = None
        self._threshold: float | None = None

    def fit(self, training_embeddings: np.ndarray) -> None:
        """Fit the OOD detector on training set embeddings.

        Args:
            training_embeddings: (n_samples, embedding_dim) array.
        """
        self._training_embeddings = training_embeddings

        if self.method == "knn":
            self._nn = NearestNeighbors(n_neighbors=self.k_neighbors, metric="cosine")
            self._nn.fit(training_embeddings)

            # Compute distances for training set (leave-one-out style)
            distances, _ = self._nn.kneighbors(training_embeddings)
            self._training_distances = distances.mean(axis=1)
        elif self.method == "mahalanobis":
            self._mean = training_embeddings.mean(axis=0)
            cov = np.cov(training_embeddings, rowvar=False)
            # Regularize covariance for numerical stability
            cov += np.eye(cov.shape[0]) * 1e-6
            self._cov_inv = np.linalg.inv(cov)

            # Compute training distances
            diffs = training_embeddings - self._mean
            self._training_distances = np.sqrt(
                np.sum(diffs @ self._cov_inv * diffs, axis=1)
            )

        self._threshold = float(np.percentile(
            self._training_distances, self.threshold_percentile
        ))
        self._fitted = True

    def score(self, embedding: np.ndarray) -> OODSignal:
        """Score a single peptide embedding for OOD-ness.

        Args:
            embedding: (embedding_dim,) array from ESM-2.

        Returns:
            OODSignal with distance, percentile, and confidence.
        """
        if not self._fitted:
            raise RuntimeError("OODDetector not fitted. Call fit() first.")

        embedding = embedding.reshape(1, -1)

        if self.method == "knn":
            distances, _ = self._nn.kneighbors(embedding)
            distance = float(distances.mean())
        elif self.method == "mahalanobis":
            diff = embedding[0] - self._mean
            distance = float(np.sqrt(diff @ self._cov_inv @ diff))

        # Percentile within training distribution
        percentile = float(
            np.mean(self._training_distances <= distance) * 100
        )

        is_ood = distance > self._threshold

        # Confidence: 1.0 if well within distribution, decreasing as distance grows
        if self._threshold > 0:
            confidence = max(0.0, 1.0 - distance / (2.0 * self._threshold))
        else:
            confidence = 0.5

        return OODSignal(
            distance=round(distance, 4),
            percentile=round(percentile, 2),
            is_ood=is_ood,
            confidence=round(confidence, 4),
        )
