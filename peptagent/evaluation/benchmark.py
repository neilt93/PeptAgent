"""Benchmark runner for comparing PeptAgent against baselines.

Runs all methods on the same targets and computes evaluation metrics.
Results are saved as JSON for downstream analysis and paper figures.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from peptagent.evaluation.metrics import (
    diversity,
    expected_calibration_error,
    hit_rate,
    novelty,
    reliability_auroc,
)

logger = logging.getLogger(__name__)


def run_benchmark(
    agent,
    baselines: dict[str, Any],
    targets: list[dict],
    ground_truth: dict[str, dict],
    output_dir: str = "results",
) -> dict:
    """Run full benchmark suite.

    Args:
        agent: The PeptAgent instance.
        baselines: Dict mapping baseline name -> baseline instance.
        targets: List of target specifications to design for.
        ground_truth: Dict mapping sequence -> {property: bool} for evaluation.
        output_dir: Where to save results.

    Returns:
        Dict with per-method metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for target in targets:
        target_name = target.get("name", "unnamed")
        logger.info(f"Benchmarking target: {target_name}")
        target_results = {}

        # Run PeptAgent (reliability-aware)
        logger.info("Running PeptAgent (reliability-aware)...")
        t0 = time.time()
        agent_candidates = agent.run(
            target_description=target["description"],
            property_requirements=target.get("properties", {}),
            constraints=target.get("constraints"),
        )
        agent_time = time.time() - t0

        agent_seqs = [c.sequence for c in agent_candidates]
        target_results["peptagent"] = {
            "candidates": [
                {
                    "sequence": c.sequence,
                    "reliability_score": c.reliability_score,
                    "properties": c.properties,
                }
                for c in agent_candidates
            ],
            "elapsed_s": round(agent_time, 2),
            "metrics": _compute_metrics(agent_candidates, ground_truth),
        }

        # Run baselines
        for baseline_name, baseline in baselines.items():
            logger.info(f"Running baseline: {baseline_name}...")
            t0 = time.time()
            if baseline_name == "random_filter":
                baseline_results = baseline.run(
                    property_thresholds=target.get("thresholds", {}),
                )
            else:
                baseline_results = baseline.run(
                    target_description=target["description"],
                )
            baseline_time = time.time() - t0

            target_results[baseline_name] = {
                "candidates": baseline_results,
                "elapsed_s": round(baseline_time, 2),
            }

        all_results[target_name] = target_results

    # Save results
    results_file = output_path / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return all_results


def _compute_metrics(candidates: list, ground_truth: dict) -> dict:
    """Compute metrics for a list of candidates."""
    sequences = [c.sequence for c in candidates]
    metrics = {
        "n_candidates": len(candidates),
        "diversity": round(diversity(sequences), 4),
        "novelty": round(novelty(sequences, set(ground_truth.keys())), 4),
    }

    # Reliability metrics (if confidence scores available)
    confidences = [c.reliability_score for c in candidates if c.reliability_score > 0]
    if confidences and ground_truth:
        is_correct = []
        conf_for_correct = []
        for c in candidates:
            if c.sequence in ground_truth and c.reliability_score > 0:
                correct = all(
                    ground_truth[c.sequence].get(prop, True)
                    for prop in c.properties
                )
                is_correct.append(correct)
                conf_for_correct.append(c.reliability_score)

        if len(set(is_correct)) == 2:
            metrics["reliability_auroc"] = round(
                reliability_auroc(conf_for_correct, is_correct), 4
            )
            metrics["ece"] = round(
                expected_calibration_error(conf_for_correct, is_correct), 4
            )

    return metrics
