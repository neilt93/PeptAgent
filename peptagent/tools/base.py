"""Base class and result type for all peptide tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Standardized return from every peptide tool.

    Attributes:
        value: The prediction, embedding, structure, or other output.
        confidence: Per-tool confidence score (0-1), None if unavailable.
        metadata: Timing, model version, raw scores, etc.
    """

    value: Any
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for inclusion in LLM context."""
        return {
            "value": _serialize(self.value),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


def _serialize(obj: Any) -> Any:
    """Best-effort serialization for LLM context."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    # numpy/torch tensors → list
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


class PeptideTool(ABC):
    """Abstract base class for peptide tools.

    Every tool takes a peptide sequence as primary input and returns a ToolResult.
    Additional arguments (e.g., target PDB for docking) are passed via kwargs.
    """

    name: str
    description: str

    @abstractmethod
    def run(self, sequence: str, **kwargs) -> ToolResult:
        """Run the tool on a single peptide sequence."""
        ...

    def batch_run(self, sequences: list[str], **kwargs) -> list[ToolResult]:
        """Run the tool on multiple sequences. Override for GPU batching."""
        return [self.run(seq, **kwargs) for seq in sequences]

    def schema(self) -> dict:
        """Return OpenAI-style function schema for LLM tool calling."""
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
                            "description": "Amino acid sequence (single-letter codes)",
                        },
                    },
                    "required": ["sequence"],
                },
            },
        }
