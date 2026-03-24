"""Conversation and candidate memory for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateRecord:
    """Record of a peptide candidate and its evaluation history."""

    sequence: str
    iteration: int
    properties: dict[str, Any] = field(default_factory=dict)
    reliability_score: float | None = None
    reliability_flags: list[str] = field(default_factory=list)
    is_ood: bool = False
    parent_sequence: str | None = None  # track lineage


class AgentMemory:
    """Track conversation history and evaluated candidates.

    Maintains both the LLM message history and a structured record of
    all candidates evaluated, enabling the agent to reference past results
    and avoid re-evaluating the same sequences.
    """

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.candidates: dict[str, CandidateRecord] = {}  # keyed by sequence
        self.iteration: int = 0

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.messages.append({"role": role, "content": content})

    def add_candidate(
        self,
        sequence: str,
        properties: dict[str, Any] | None = None,
        reliability_score: float | None = None,
        reliability_flags: list[str] | None = None,
        is_ood: bool = False,
        parent: str | None = None,
    ) -> CandidateRecord:
        """Record a candidate evaluation."""
        if sequence in self.candidates:
            # Update existing record
            record = self.candidates[sequence]
            if properties:
                record.properties.update(properties)
            if reliability_score is not None:
                record.reliability_score = reliability_score
            if reliability_flags:
                record.reliability_flags = reliability_flags
            record.is_ood = is_ood
        else:
            record = CandidateRecord(
                sequence=sequence,
                iteration=self.iteration,
                properties=properties or {},
                reliability_score=reliability_score,
                reliability_flags=reliability_flags or [],
                is_ood=is_ood,
                parent_sequence=parent,
            )
            self.candidates[sequence] = record
        return record

    def was_evaluated(self, sequence: str) -> bool:
        """Check if a sequence has already been evaluated."""
        return sequence in self.candidates

    def get_top_candidates(self, n: int = 5, min_confidence: float = 0.0) -> list[CandidateRecord]:
        """Get top candidates by reliability score."""
        scored = [
            c for c in self.candidates.values()
            if c.reliability_score is not None and c.reliability_score >= min_confidence
        ]
        scored.sort(key=lambda c: c.reliability_score or 0, reverse=True)
        return scored[:n]

    def get_context_summary(self) -> str:
        """Summarize evaluated candidates for LLM context."""
        if not self.candidates:
            return "No candidates evaluated yet."

        lines = [f"EVALUATED CANDIDATES ({len(self.candidates)} total):"]
        for seq, record in self.candidates.items():
            conf_str = f"{record.reliability_score:.2f}" if record.reliability_score else "N/A"
            flag_str = f" [{', '.join(record.reliability_flags)}]" if record.reliability_flags else ""
            lines.append(f"  {seq[:20]}... | confidence={conf_str}{flag_str}")
        return "\n".join(lines)

    def advance_iteration(self) -> None:
        """Move to the next iteration."""
        self.iteration += 1
