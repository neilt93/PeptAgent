"""Train/calibration/test splitting for peptide datasets.

The three-way split is important: train for model fitting, calibration
for fitting the reliability layer, and test for final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from peptagent.data.datasets import PeptideDataset


@dataclass
class DataSplit:
    """Three-way split of a peptide dataset."""

    train: PeptideDataset
    calibration: PeptideDataset
    test: PeptideDataset


def three_way_split(
    dataset: PeptideDataset,
    train_frac: float = 0.7,
    cal_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> DataSplit:
    """Split dataset into train/calibration/test sets."""
    assert abs(train_frac + cal_frac + test_frac - 1.0) < 1e-6

    n = len(dataset)
    indices = np.arange(n)

    # First split: train vs (cal + test)
    train_idx, rest_idx = train_test_split(
        indices, train_size=train_frac, random_state=seed
    )

    # Second split: cal vs test
    relative_cal = cal_frac / (cal_frac + test_frac)
    cal_idx, test_idx = train_test_split(
        rest_idx, train_size=relative_cal, random_state=seed
    )

    return DataSplit(
        train=_subset(dataset, train_idx, "train"),
        calibration=_subset(dataset, cal_idx, "calibration"),
        test=_subset(dataset, test_idx, "test"),
    )


def _subset(dataset: PeptideDataset, indices: np.ndarray, split_name: str) -> PeptideDataset:
    """Create a subset of a dataset from indices."""
    return PeptideDataset(
        name=f"{dataset.name}_{split_name}",
        sequences=[dataset.sequences[i] for i in indices],
        labels={
            prop: [vals[i] for i in indices]
            for prop, vals in dataset.labels.items()
        },
        metadata={**dataset.metadata, "split": split_name, "n_samples": len(indices)},
    )
