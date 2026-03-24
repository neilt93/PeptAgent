"""Baseline methods for comparison against PeptAgent.

Two baselines:
1. RandomFilter: Generate random sequences, filter by property predictions
2. SingleShot: Single-pass LLM generation without iterative refinement
"""

from __future__ import annotations

import random
from typing import Any

from peptagent.tools.base import ToolResult
from peptagent.tools.generator import AMINO_ACIDS
from peptagent.tools.registry import ToolRegistry


class RandomFilterBaseline:
    """Generate random peptides and filter by property predictions."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tools = tool_registry

    def run(
        self,
        n_generate: int = 1000,
        n_select: int = 10,
        min_length: int = 8,
        max_length: int = 30,
        property_thresholds: dict[str, float] | None = None,
    ) -> list[dict]:
        """Generate random sequences and filter by property predictions."""
        property_thresholds = property_thresholds or {}

        candidates = []
        for _ in range(n_generate):
            length = random.randint(min_length, max_length)
            seq = "".join(random.choice(AMINO_ACIDS) for _ in range(length))
            candidates.append(seq)

        scored = []
        for seq in candidates:
            try:
                result = self.tools.run_tool("predict_all_properties", seq)
                if isinstance(result.value, dict):
                    passes = True
                    for prop, threshold in property_thresholds.items():
                        if prop in result.value:
                            prob = result.value[prop].get("probability", 0)
                            if prob < threshold:
                                passes = False
                                break
                    if passes:
                        scored.append({"sequence": seq, "properties": result.value})
            except Exception:
                continue

        return scored[:n_select]


class SingleShotBaseline:
    """Single-pass LLM generation without iterative refinement."""

    def __init__(self, llm_client: Any, tool_registry: ToolRegistry) -> None:
        self.llm = llm_client
        self.tools = tool_registry

    def run(
        self,
        target_description: str,
        n_candidates: int = 10,
    ) -> list[dict]:
        """Generate and evaluate in a single pass."""
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a peptide design expert.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Generate {n_candidates} therapeutic peptide sequences for: "
                        f"{target_description}\n\n"
                        "Return ONLY sequences, one per line, single-letter amino acid codes."
                    ),
                },
            ],
            temperature=0.8,
        )

        text = response.choices[0].message.content
        sequences = []
        for line in text.strip().split("\n"):
            clean = line.strip().lstrip("0123456789.-) ")
            if clean and all(c in AMINO_ACIDS for c in clean) and 5 <= len(clean) <= 50:
                sequences.append(clean)

        results = []
        for seq in sequences[:n_candidates]:
            try:
                result = self.tools.run_tool("predict_all_properties", seq)
                results.append({"sequence": seq, "properties": result.value})
            except Exception:
                results.append({"sequence": seq, "properties": {}})

        return results
