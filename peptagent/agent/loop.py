"""Core agent loop for PeptAgent.

This is the main contribution of the paper — an LLM-orchestrated loop that
iteratively designs peptides using tool calls and reliability-aware decision
making. The loop is intentionally simple (~100 lines of core logic) with no
framework dependencies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from peptagent.agent.memory import AgentMemory
from peptagent.agent.prompts import (
    EVALUATION_RESULT_TEMPLATE,
    EVALUATION_RESULT_TEMPLATE_NO_RELIABILITY,
    ITERATION_PROMPT,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NO_RELIABILITY,
    TASK_TEMPLATE,
)
from peptagent.reliability.aggregator import ReliabilityAggregator
from peptagent.tools.base import ToolResult
from peptagent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A fully evaluated peptide candidate."""

    sequence: str
    properties: dict[str, Any]
    reliability_score: float
    reliability_flags: list[str] = field(default_factory=list)
    iteration: int = 0


class AgentLoop:
    """LLM-orchestrated peptide design loop.

    The agent takes a target specification, generates candidates,
    evaluates them through the tool chain, assesses reliability,
    and iteratively refines until convergence or max iterations.

    The reliability_aware flag controls whether the agent receives
    reliability signals — this is the key ablation for the paper.
    """

    def __init__(
        self,
        llm_client: Any,
        tool_registry: ToolRegistry,
        reliability: ReliabilityAggregator | None = None,
        config: Any = None,
        reliability_aware: bool = True,
    ) -> None:
        self.llm = llm_client
        self.tools = tool_registry
        self.reliability = reliability
        self.config = config or {}
        self.reliability_aware = reliability_aware
        self.memory = AgentMemory()

        # Agent config
        agent_cfg = config.get("agent", {}) if config else {}
        self.max_iterations = agent_cfg.get("max_iterations", 10)
        self.early_stop_confidence = agent_cfg.get("early_stop_confidence", 0.85)

    def run(
        self,
        target_description: str,
        property_requirements: dict[str, str],
        constraints: dict[str, Any] | None = None,
    ) -> list[Candidate]:
        """Run the full agent loop.

        Args:
            target_description: Natural language description of the target
                (e.g., \"antimicrobial peptide targeting gram-negative bacteria\").
            property_requirements: Desired property profile
                (e.g., {\"hemolysis\": \"low\", \"solubility\": \"high\"}).
            constraints: Design constraints
                (e.g., {\"min_length\": 10, \"max_length\": 30}).

        Returns:
            List of Candidate objects, ranked by reliability score.
        """
        constraints = constraints or {}
        system_prompt = SYSTEM_PROMPT if self.reliability_aware else SYSTEM_PROMPT_NO_RELIABILITY

        # Initialize conversation
        self.memory.add_message("system", system_prompt)

        task = TASK_TEMPLATE.format(
            target_description=target_description,
            property_requirements="\n".join(f"  - {k}: {v}" for k, v in property_requirements.items()),
            constraints="\n".join(f"  - {k}: {v}" for k, v in constraints.items()) or "  None",
        )
        self.memory.add_message("user", task)

        # Main loop
        for iteration in range(self.max_iterations):
            self.memory.advance_iteration()
            logger.info(f"=== Iteration {iteration + 1}/{self.max_iterations} ===")

            # Get LLM response (with tool calling)
            response = self._call_llm()

            if response is None:
                logger.warning("LLM returned no response, stopping")
                break

            # Process tool calls if any
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_results = self._execute_tool_calls(response.tool_calls)
                self._process_results(tool_results, iteration)
            else:
                # LLM chose to respond without tools — likely presenting final results
                self.memory.add_message("assistant", response.content)

            # Check early stopping
            top = self.memory.get_top_candidates(n=1)
            if top and top[0].reliability_score and top[0].reliability_score >= self.early_stop_confidence:
                logger.info(
                    f"Early stop: top candidate confidence "
                    f"{top[0].reliability_score:.2f} >= {self.early_stop_confidence}"
                )
                break

            # Prompt for next iteration
            self.memory.add_message("user", ITERATION_PROMPT)

        # Return ranked candidates
        return self._compile_results()

    def _call_llm(self) -> Any:
        """Call the LLM with current conversation and tool schemas."""
        try:
            response = self.llm.chat.completions.create(
                model=self.config.get("llm", {}).get("model", "gpt-4o"),
                messages=self.memory.messages,
                tools=self.tools.list_schemas(),
                tool_choice="auto",
                temperature=self.config.get("llm", {}).get("temperature", 0.7),
                max_tokens=self.config.get("llm", {}).get("max_tokens", 4096),
            )
            return response.choices[0].message
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _execute_tool_calls(self, tool_calls: list) -> dict[str, list[ToolResult]]:
        """Execute tool calls requested by the LLM."""
        results: dict[str, list[ToolResult]] = {}

        for call in tool_calls:
            tool_name = call.function.name
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse tool args: {call.function.arguments}")
                continue

            sequence = args.pop("sequence", "")
            logger.info(f"Executing tool: {tool_name}({sequence[:20]}...)")

            try:
                result = self.tools.run_tool(tool_name, sequence, **args)
                if sequence not in results:
                    results[sequence] = []
                results[sequence].append(result)
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")

        return results

    def _process_results(
        self, tool_results: dict[str, list[ToolResult]], iteration: int
    ) -> None:
        """Process tool results, compute reliability, and update context."""
        for sequence, results in tool_results.items():
            # Aggregate property results
            property_results = {}
            for result in results:
                if isinstance(result.value, dict) and "probability" in result.value:
                    prop_name = result.metadata.get("property", "unknown")
                    property_results[prop_name] = result

            # Compute reliability if enabled
            reliability_report = None
            if self.reliability_aware and self.reliability and property_results:
                # Try to get embedding for OOD check
                embedding = None
                if "esm2_embed" in self.tools.list_names():
                    try:
                        embed_result = self.tools.run_tool("esm2_embed", sequence)
                        embedding = embed_result.value.numpy() if hasattr(embed_result.value, "numpy") else np.array(embed_result.value)
                    except Exception:
                        pass

                reliability_report = self.reliability.score(
                    sequence, property_results, embedding
                )

                # Record candidate with reliability info
                self.memory.add_candidate(
                    sequence=sequence,
                    properties={k: r.to_dict() for k, r in property_results.items()},
                    reliability_score=reliability_report.overall_confidence,
                    reliability_flags=reliability_report.flags,
                    is_ood=reliability_report.is_ood,
                )
            else:
                # Without reliability
                self.memory.add_candidate(
                    sequence=sequence,
                    properties={k: r.to_dict() for k, r in property_results.items()},
                )

            # Format results for LLM context
            props_str = "\n".join(
                f"  {k}: {json.dumps(r.to_dict()['value'])}"
                for k, r in property_results.items()
            )

            if self.reliability_aware and reliability_report:
                context = EVALUATION_RESULT_TEMPLATE.format(
                    sequence=sequence,
                    property_results=props_str,
                    reliability_report=reliability_report.to_context_string(),
                )
            else:
                context = EVALUATION_RESULT_TEMPLATE_NO_RELIABILITY.format(
                    sequence=sequence,
                    property_results=props_str,
                )

            self.memory.add_message("tool", context)

    def _compile_results(self) -> list[Candidate]:
        """Compile final ranked candidate list."""
        candidates = []
        for record in self.memory.candidates.values():
            candidates.append(
                Candidate(
                    sequence=record.sequence,
                    properties=record.properties,
                    reliability_score=record.reliability_score or 0.0,
                    reliability_flags=record.reliability_flags,
                    iteration=record.iteration,
                )
            )
        candidates.sort(key=lambda c: c.reliability_score, reverse=True)
        return candidates
