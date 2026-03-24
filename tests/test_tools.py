"""Tests for peptide tools."""

import pytest

from peptagent.tools.base import PeptideTool, ToolResult
from peptagent.tools.registry import ToolRegistry


class MockTool(PeptideTool):
    name = "mock_tool"
    description = "A mock tool for testing"

    def run(self, sequence, **kwargs):
        return ToolResult(
            value={"length": len(sequence)},
            confidence=0.95,
            metadata={"mock": True},
        )


class TestToolResult:
    def test_to_dict(self):
        result = ToolResult(value={"score": 0.8}, confidence=0.9, metadata={"model": "test"})
        d = result.to_dict()
        assert d["value"] == {"score": 0.8}
        assert d["confidence"] == 0.9

    def test_serialize_numpy(self):
        import numpy as np
        result = ToolResult(value=np.array([1.0, 2.0, 3.0]))
        d = result.to_dict()
        assert d["value"] == [1.0, 2.0, 3.0]


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        assert registry.get("mock_tool") is tool

    def test_duplicate_registration_raises(self):
        registry = ToolRegistry()
        registry.register(MockTool())
        with pytest.raises(ValueError):
            registry.register(MockTool())

    def test_missing_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_schemas(self):
        registry = ToolRegistry()
        registry.register(MockTool())
        schemas = registry.list_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "mock_tool"

    def test_run_tool(self):
        registry = ToolRegistry()
        registry.register(MockTool())
        result = registry.run_tool("mock_tool", "ACDEFGHIK")
        assert result.value["length"] == 9
        assert result.confidence == 0.95
