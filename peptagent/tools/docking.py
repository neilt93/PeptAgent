"""AutoDock Vina docking tool for peptide-protein interactions."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from peptagent.tools.base import PeptideTool, ToolResult

logger = logging.getLogger(__name__)


class VinaDocking(PeptideTool):
    """Dock a peptide against a target protein using AutoDock Vina.

    Requires a predicted or known peptide structure (PDB) and a target
    protein structure. The agent should call predict_structure first
    to obtain the peptide PDB.
    """

    name = "dock_peptide"
    description = (
        "Dock a peptide against a target protein to estimate binding affinity. "
        "Requires 'target_pdb' (path to target protein PDB) and optionally "
        "'peptide_pdb' (PDB string of peptide structure). Returns binding "
        "energy in kcal/mol (more negative = stronger binding)."
    )

    def __init__(
        self,
        exhaustiveness: int = 8,
        num_modes: int = 5,
    ) -> None:
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes

    def run(self, sequence: str, **kwargs: Any) -> ToolResult:
        target_pdb = kwargs.get("target_pdb")
        peptide_pdb = kwargs.get("peptide_pdb")

        if target_pdb is None:
            return ToolResult(
                value={"error": "target_pdb is required for docking"},
                confidence=0.0,
                metadata={},
            )

        t0 = time.time()

        try:
            from vina import Vina

            v = Vina(sf_name="vina")

            # Prepare receptor
            v.set_receptor(target_pdb)

            # Write peptide PDB to temp file and prepare ligand
            with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
                f.write(peptide_pdb)
                peptide_pdb_path = f.name

            ligand_pdbqt = self._pdb_to_pdbqt(peptide_pdb_path)
            v.set_ligand_from_file(ligand_pdbqt)

            # Compute box around receptor binding site
            center, size = self._compute_search_box(target_pdb)
            v.compute_vina_maps(center=center, box_size=size)

            # Dock
            v.dock(exhaustiveness=self.exhaustiveness, n_poses=self.num_modes)
            energies = v.energies()

            best_energy = float(energies[0][0])

            # Rough confidence: very strong binding (<-10) is high confidence,
            # weak binding (>-3) is low confidence
            confidence = min(1.0, max(0.0, (-best_energy - 3.0) / 7.0))

            return ToolResult(
                value={
                    "best_binding_energy": round(best_energy, 2),
                    "all_energies": [round(float(e[0]), 2) for e in energies],
                    "num_poses": len(energies),
                },
                confidence=round(confidence, 4),
                metadata={
                    "exhaustiveness": self.exhaustiveness,
                    "elapsed_s": round(time.time() - t0, 3),
                },
            )

        except ImportError:
            logger.warning("Vina not installed, returning placeholder result")
            return ToolResult(
                value={"error": "vina package not installed"},
                confidence=0.0,
                metadata={"elapsed_s": round(time.time() - t0, 3)},
            )

    @staticmethod
    def _pdb_to_pdbqt(pdb_path: str) -> str:
        """Convert PDB to PDBQT format using meeko."""
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            from rdkit import Chem

            mol = Chem.MolFromPDBFile(pdb_path)
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            pdbqt_path = pdb_path.replace(".pdb", ".pdbqt")
            for setup in mol_setups:
                pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
                if is_ok:
                    with open(pdbqt_path, "w") as f:
                        f.write(pdbqt_string)
                    return pdbqt_path
        except ImportError:
            pass
        # Fallback: return PDB path (Vina can sometimes handle it)
        return pdb_path

    @staticmethod
    def _compute_search_box(target_pdb: str) -> tuple[list[float], list[float]]:
        """Compute search box centered on the target protein."""
        try:
            from Bio.PDB import PDBParser

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("target", target_pdb)
            coords = [atom.get_vector().get_array() for atom in structure.get_atoms()]
            import numpy as np

            coords = np.array(coords)
            center = coords.mean(axis=0).tolist()
            size = ((coords.max(axis=0) - coords.min(axis=0)) + 10).tolist()  # 10A padding
            return center, size
        except Exception:
            # Default box
            return [0.0, 0.0, 0.0], [30.0, 30.0, 30.0]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sequence": {
                            "type": "string",
                            "description": "Amino acid sequence of the peptide",
                        },
                        "target_pdb": {
                            "type": "string",
                            "description": "Path to the target protein PDB file",
                        },
                        "peptide_pdb": {
                            "type": "string",
                            "description": "PDB string of the peptide structure",
                        },
                    },
                    "required": ["sequence", "target_pdb"],
                },
            },
        }
