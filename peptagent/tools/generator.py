"""Peptide sequence generation tools."""

from __future__ import annotations

import random
import re
import time
from typing import Any

from peptagent.tools.base import PeptideTool, ToolResult

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class LLMGuidedGenerator(PeptideTool):
    """Generate peptide sequences using LLM-guided design.

    The LLM proposes sequences based on target properties and design
    constraints. This is a tool the agent calls to generate candidates,
    distinct from the agent's own reasoning.
    """

    name = "generate_peptides"
    description = (
        "Generate candidate peptide sequences given a design specification. "
        "Accepts 'target_properties' (desired property profile), 'num_candidates' "
        "(how many to generate, default 5), and 'constraints' (length, required "
        "motifs, etc.). Returns a list of candidate sequences."
    )

    def __init__(self, llm_client: Any = None, method: str = "llm_guided") -> None:
        self.llm_client = llm_client
        self.method = method

    def run(self, sequence: str = "", **kwargs: Any) -> ToolResult:
        """Generate candidates. 'sequence' can be a seed/template or empty."""
        t0 = time.time()
        num_candidates = kwargs.get("num_candidates", 5)
        target_properties = kwargs.get("target_properties", {})
        constraints = kwargs.get("constraints", {})

        if self.method == "llm_guided" and self.llm_client is not None:
            candidates = self._llm_generate(
                seed=sequence,
                target_properties=target_properties,
                constraints=constraints,
                n=num_candidates,
            )
        else:
            candidates = self._random_generate(
                seed=sequence,
                constraints=constraints,
                n=num_candidates,
            )

        return ToolResult(
            value={"candidates": candidates, "num_generated": len(candidates)},
            confidence=None,
            metadata={
                "method": self.method,
                "seed": sequence if sequence else None,
                "elapsed_s": round(time.time() - t0, 3),
            },
        )

    def _llm_generate(
        self,
        seed: str,
        target_properties: dict,
        constraints: dict,
        n: int,
    ) -> list[str]:
        """Use the LLM to propose peptide sequences."""
        min_len = constraints.get("min_length", 5)
        max_len = constraints.get("max_length", 50)

        prompt = (
            f"Generate {n} therapeutic peptide sequences with these properties:\n"
            f"Target properties: {target_properties}\n"
            f"Length: {min_len}-{max_len} amino acids\n"
        )
        if seed:
            prompt += f"Base/template sequence: {seed}\n"
        if constraints.get("required_motifs"):
            prompt += f"Must contain motifs: {constraints['required_motifs']}\n"
        prompt += (
            "\nReturn ONLY the sequences, one per line, using single-letter amino acid codes. "
            "No numbering, no explanations."
        )

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a peptide design expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        )

        text = response.choices[0].message.content
        candidates = self._parse_sequences(text, min_len, max_len)
        return candidates[:n]

    def _random_generate(self, seed: str, constraints: dict, n: int) -> list[str]:
        """Random generation with optional seed mutation. Baseline method."""
        min_len = constraints.get("min_length", 8)
        max_len = constraints.get("max_length", 30)
        candidates = []

        for _ in range(n):
            if seed:
                # Mutate seed: substitute 1-3 random positions
                seq = list(seed)
                n_mutations = random.randint(1, min(3, len(seq)))
                for _ in range(n_mutations):
                    pos = random.randint(0, len(seq) - 1)
                    seq[pos] = random.choice(AMINO_ACIDS)
                candidates.append("".join(seq))
            else:
                length = random.randint(min_len, max_len)
                seq = "".join(random.choice(AMINO_ACIDS) for _ in range(length))
                candidates.append(seq)

        return candidates

    @staticmethod
    def _parse_sequences(text: str, min_len: int, max_len: int) -> list[str]:
        """Extract valid amino acid sequences from LLM output."""
        sequences = []
        for line in text.strip().split("\n"):
            # Strip numbering, bullets, etc.
            clean = re.sub(r"^[\d\.\)\-\*\s]+", "", line).strip()
            # Validate: only standard amino acid letters
            if clean and all(c in AMINO_ACIDS for c in clean):
                if min_len <= len(clean) <= max_len:
                    sequences.append(clean)
        return sequences

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
                            "description": "Seed/template sequence (optional, can be empty)",
                        },
                        "num_candidates": {
                            "type": "integer",
                            "description": "Number of candidates to generate (default 5)",
                        },
                        "target_properties": {
                            "type": "object",
                            "description": "Desired property profile (e.g., {hemolysis: false, solubility: high})",
                        },
                        "constraints": {
                            "type": "object",
                            "description": "Design constraints (min_length, max_length, required_motifs)",
                        },
                    },
                    "required": [],
                },
            },
        }
