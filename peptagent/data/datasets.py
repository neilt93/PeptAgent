"""Dataset loaders for peptide benchmarks.

Supports DBAASP (antimicrobial peptides) and SATPdb (therapeutic peptides).
Each dataset returns sequences with experimentally validated properties.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PeptideDataset:
    """A dataset of peptide sequences with property labels."""

    name: str
    sequences: list[str]
    labels: dict[str, list[bool | float]]  # property_name -> labels
    metadata: dict[str, Any]

    def __len__(self) -> int:
        return len(self.sequences)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({"sequence": self.sequences})
        for prop, vals in self.labels.items():
            df[prop] = vals
        return df


def load_dbaasp(data_dir: str = "data/dbaasp") -> PeptideDataset:
    """Load DBAASP antimicrobial peptide dataset."""
    path = Path(data_dir)
    sequences = []
    labels: dict[str, list] = {"antimicrobial": [], "hemolytic": []}

    csv_path = path / "dbaasp_peptides.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        sequences = df["sequence"].tolist()
        if "activity" in df.columns:
            labels["antimicrobial"] = (df["activity"] == "active").tolist()
        if "hemolytic" in df.columns:
            labels["hemolytic"] = df["hemolytic"].tolist()
    else:
        logger.warning(f"DBAASP data not found at {csv_path}. Use download script first.")

    return PeptideDataset(
        name="dbaasp",
        sequences=sequences,
        labels=labels,
        metadata={"source": "DBAASP", "url": "https://dbaasp.org"},
    )


def load_satpdb(data_dir: str = "data/satpdb") -> PeptideDataset:
    """Load SATPdb therapeutic peptide dataset."""
    path = Path(data_dir)
    sequences = []
    labels: dict[str, list] = {}

    csv_path = path / "satpdb_peptides.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        sequences = df["sequence"].tolist()
        for col in df.columns:
            if col != "sequence":
                labels[col] = df[col].tolist()
    else:
        logger.warning(f"SATPdb data not found at {csv_path}. Use download script first.")

    return PeptideDataset(
        name="satpdb",
        sequences=sequences,
        labels=labels,
        metadata={"source": "SATPdb", "url": "https://webs.iiitd.edu.in/raghava/satpdb/"},
    )


DATASET_REGISTRY = {
    "dbaasp_amp": load_dbaasp,
    "satpdb": load_satpdb,
}


def load_dataset(name: str, **kwargs: Any) -> PeptideDataset:
    """Load a dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](**kwargs)
