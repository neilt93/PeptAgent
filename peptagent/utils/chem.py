"""Amino acid and peptide chemistry utilities."""

from __future__ import annotations

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Average molecular weights (Da)
AA_WEIGHTS = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.03, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}

WATER_MASS = 18.02


def is_valid_sequence(sequence: str) -> bool:
    """Check if a sequence contains only standard amino acids."""
    return bool(sequence) and all(c in STANDARD_AA for c in sequence)


def molecular_weight(sequence: str) -> float:
    """Calculate molecular weight of a peptide (Da)."""
    if not sequence:
        return 0.0
    return sum(AA_WEIGHTS.get(aa, 0) for aa in sequence) - (len(sequence) - 1) * WATER_MASS


def charge_at_ph(sequence: str, ph: float = 7.0) -> float:
    """Estimate net charge at given pH using Henderson-Hasselbalch."""
    pka = {
        "K": 10.5, "R": 12.5, "H": 6.0,   # positive
        "D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1,  # negative
    }
    # N-terminal and C-terminal
    charge = 1.0 / (1.0 + 10 ** (ph - 9.69))  # N-term pKa ~9.69
    charge -= 1.0 / (1.0 + 10 ** (2.34 - ph))  # C-term pKa ~2.34

    for aa in sequence:
        if aa in ("K", "R", "H"):
            charge += 1.0 / (1.0 + 10 ** (ph - pka[aa]))
        elif aa in ("D", "E", "C", "Y"):
            charge -= 1.0 / (1.0 + 10 ** (pka[aa] - ph))

    return round(charge, 2)


def hydrophobicity(sequence: str) -> float:
    """Average Kyte-Doolittle hydrophobicity index."""
    kd = {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
        "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
        "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
        "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    }
    if not sequence:
        return 0.0
    return round(sum(kd.get(aa, 0) for aa in sequence) / len(sequence), 3)
