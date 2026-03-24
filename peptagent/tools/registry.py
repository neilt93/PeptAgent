"""Tool registry: register, lookup, and dispatch peptide tools."""

from __future__ import annotations

import logging
from typing import Any

from peptagent.tools.base import PeptideTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all available peptide tools.

    Tools are registered at startup based on config. The registry generates
    function schemas for the LLM and dispatches tool calls by name.
    """

    def __init__(self) -> None:
        self._tools: dict[str, PeptideTool] = {}

    def register(self, tool: PeptideTool) -> None:
        """Register a tool by its name."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> PeptideTool:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def list_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_schemas(self) -> list[dict]:
        """Generate OpenAI-style function schemas for all registered tools."""
        return [tool.schema() for tool in self._tools.values()]

    def run_tool(self, name: str, sequence: str, **kwargs: Any) -> ToolResult:
        """Dispatch a tool call by name."""
        tool = self.get(name)
        logger.info(f"Running tool '{name}' on sequence: {sequence[:20]}...")
        result = tool.run(sequence, **kwargs)
        logger.info(f"Tool '{name}' returned confidence={result.confidence}")
        return result

    def run_tool_batch(self, name: str, sequences: list[str], **kwargs: Any) -> list[ToolResult]:
        """Dispatch a batch tool call by name."""
        tool = self.get(name)
        return tool.batch_run(sequences, **kwargs)
