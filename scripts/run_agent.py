"""CLI entry point for running PeptAgent."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from peptagent.agent.loop import AgentLoop
from peptagent.reliability.aggregator import ReliabilityAggregator
from peptagent.reliability.calibration import Calibrator
from peptagent.reliability.ensemble import EnsembleEstimator
from peptagent.reliability.ood import OODDetector
from peptagent.tools.registry import ToolRegistry
from peptagent.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run PeptAgent peptide design")
    parser.add_argument("--config", type=str, help="Path to experiment config YAML")
    parser.add_argument("--target", type=str, required=True, help="Target description")
    parser.add_argument(
        "--properties", type=str, nargs="+", default=["hemolysis:low", "solubility:high"],
        help="Property requirements as key:value pairs",
    )
    parser.add_argument("--no-reliability", action="store_true", help="Disable reliability layer (ablation)")
    parser.add_argument("--output", type=str, default="results/agent_output.json", help="Output file")
    parser.add_argument(
        "--override", type=str, nargs="*", default=[], help="Config overrides (dotlist format)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    cfg = load_config(args.config, args.override)

    # Parse property requirements
    prop_reqs = {}
    for p in args.properties:
        key, val = p.split(":", 1)
        prop_reqs[key] = val

    # Initialize tools
    registry = ToolRegistry()
    # TODO: Register tools based on config (Phase 1-2 implementation)

    # Initialize reliability layer
    reliability = None
    if not args.no_reliability:
        reliability = ReliabilityAggregator(
            ensemble=EnsembleEstimator(
                disagreement_threshold=cfg.reliability.ensemble.disagreement_threshold,
            ),
            ood=OODDetector(
                method=cfg.reliability.ood.method,
                threshold_percentile=cfg.reliability.ood.threshold * 100,
            ),
        )

    # Initialize LLM client
    if cfg.llm.provider == "openai":
        from openai import OpenAI
        llm_client = OpenAI()
    elif cfg.llm.provider == "anthropic":
        from anthropic import Anthropic
        llm_client = Anthropic()
    else:
        raise ValueError(f"Unknown LLM provider: {cfg.llm.provider}")

    # Run agent
    agent = AgentLoop(
        llm_client=llm_client,
        tool_registry=registry,
        reliability=reliability,
        config=cfg,
        reliability_aware=not args.no_reliability,
    )

    candidates = agent.run(
        target_description=args.target,
        property_requirements=prop_reqs,
    )

    # Save results
    results = {
        "target": args.target,
        "property_requirements": prop_reqs,
        "reliability_aware": not args.no_reliability,
        "candidates": [
            {
                "sequence": c.sequence,
                "reliability_score": c.reliability_score,
                "properties": c.properties,
                "reliability_flags": c.reliability_flags,
                "iteration": c.iteration,
            }
            for c in candidates
        ],
    }

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDesigned {len(candidates)} candidates. Results saved to {args.output}")
    for i, c in enumerate(candidates[:5]):
        print(f"  {i+1}. {c.sequence} (confidence: {c.reliability_score:.2f})")


if __name__ == "__main__":
    main()
