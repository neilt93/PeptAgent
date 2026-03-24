"""ESMFold structure prediction tool."""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

from peptagent.tools.base import PeptideTool, ToolResult


class ESMFoldPredictor(PeptideTool):
    """Predict 3D structure from peptide sequence using ESMFold.

    Returns predicted structure as PDB string along with per-residue
    confidence (pLDDT) scores.
    """

    name = "predict_structure"
    description = (
        "Predict the 3D structure of a peptide sequence using ESMFold. "
        "Returns a PDB string and per-residue confidence (pLDDT) scores. "
        "Use this before docking or when structural features matter."
    )

    def __init__(self, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def run(self, sequence: str, **kwargs: Any) -> ToolResult:
        t0 = time.time()
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        # Extract pLDDT confidence scores
        plddt = outputs["plddt"][0, :, 1].cpu().numpy()
        mean_plddt = float(plddt.mean())

        # Convert to PDB
        pdb_string = self._output_to_pdb(outputs, sequence)

        # pLDDT is 0-100, normalize to 0-1 for confidence
        confidence = mean_plddt / 100.0

        return ToolResult(
            value={
                "pdb": pdb_string,
                "mean_plddt": round(mean_plddt, 2),
                "per_residue_plddt": [round(float(x), 2) for x in plddt],
            },
            confidence=round(confidence, 4),
            metadata={
                "model": "esmfold_v1",
                "sequence_length": len(sequence),
                "elapsed_s": round(time.time() - t0, 3),
            },
        )

    @staticmethod
    def _output_to_pdb(outputs: Any, sequence: str) -> str:
        """Convert ESMFold output to PDB format string."""
        # Use the built-in conversion from transformers if available
        try:
            from transformers.models.esm.openfold_utils.protein import Protein, to_pdb

            protein = Protein(
                aatype=outputs["aatype"][0].cpu().numpy(),
                atom_positions=outputs["positions"][-1, 0].cpu().numpy(),
                atom_mask=outputs["atom37_atom_exists"][0].cpu().numpy(),
                residue_index=(outputs["residue_index"][0] + 1).cpu().numpy(),
                b_factors=outputs["plddt"][0].cpu().numpy(),
                chain_index=outputs["chain_index"][0].cpu().numpy() if "chain_index" in outputs else None,
            )
            return to_pdb(protein)
        except ImportError:
            return f"REMARK PDB conversion unavailable. Sequence: {sequence}"
