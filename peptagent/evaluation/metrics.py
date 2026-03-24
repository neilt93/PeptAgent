"""Evaluation metrics for peptide design quality and reliability.

Metrics cover both design quality (hit rate, diversity, novelty) and
reliability estimation (calibration error, selective prediction AUROC).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def hit_rate(predictions: list[bool], ground_truth: list[bool]) -> float:
    """Fraction of predicted-active peptides that are truly active."""
    if not predictions:
        return 0.0
    hits = sum(p and g for p, g in zip(predictions, ground_truth))
    predicted_positive = sum(predictions)
    return hits / predicted_positive if predicted_positive > 0 else 0.0


def diversity(sequences: list[str]) -> float:
    """Average pairwise sequence distance (normalized edit distance).

    Higher diversity means the agent explored more of sequence space.
    """
    if len(sequences) < 2:
        return 0.0

    distances = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            d = _normalized_edit_distance(sequences[i], sequences[j])
            distances.append(d)
    return float(np.mean(distances))


def novelty(generated: list[str], reference: set[str]) -> float:
    """Fraction of generated sequences not in the reference set."""
    if not generated:
        return 0.0
    novel = sum(1 for s in generated if s not in reference)
    return novel / len(generated)


def reliability_auroc(
    confidences: list[float],
    is_correct: list[bool],
) -> float:
    """AUROC for reliability estimation: can the confidence score distinguish
    correct from incorrect predictions?

    Higher AUROC means the reliability layer is well-calibrated.
    """
    if len(set(is_correct)) < 2:
        return float("nan")  # need both classes
    return float(roc_auc_score(is_correct, confidences))


def expected_calibration_error(
    confidences: list[float],
    accuracies: list[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Measures how well confidence scores match actual accuracy.
    Lower is better.
    """
    confidences = np.array(confidences)
    accuracies = np.array(accuracies, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)

    return float(ece)


def _normalized_edit_distance(s1: str, s2: str) -> float:
    """Levenshtein distance normalized by max length."""
    if not s1 and not s2:
        return 0.0
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / max(m, n)
