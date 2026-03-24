# PeptAgent

**LLM-orchestrated autonomous therapeutic peptide design with built-in reliability estimation.**

PeptAgent is the first agentic system purpose-built for therapeutic peptide design. Unlike existing agentic drug discovery frameworks (LIDDiA, Prompt-to-Pill, ChemAgents) which target small molecules, PeptAgent chains peptide-specific computational tools and includes uncertainty-aware decision making to flag and handle unreliable predictions during the design loop.

## Architecture

```
Target Specification ──► LLM Orchestrator ──► Tool Dispatch ──► Evaluation
        ▲                      │                    │               │
        │                      ▼                    ▼               ▼
        │                  Reasoning            ESM-2           Properties
        │                  + Planning           ESMFold         Structure
        │                      │                Vina            Docking
        │                      ▼                    │               │
        └── Reliability ◄── Aggregator ◄───── Confidence ◄─── Results
            Report              │
                                ▼
                          Agent Decision
                     (refine / generate / stop)
```

### Key Components

- **Agent Loop** (`peptagent/agent/`): LLM-orchestrated iterative design. No framework dependencies — the loop is ~100 lines of core logic.
- **Tool Registry** (`peptagent/tools/`): Wraps peptide-specific models (ESM-2, ESMFold, property predictors, AutoDock Vina) as callable functions with standardized I/O.
- **Reliability Layer** (`peptagent/reliability/`): Ensemble disagreement, OOD detection (Mahalanobis/kNN in ESM-2 space), and post-hoc calibration. Feeds confidence signals back to the agent.
- **Evaluation** (`peptagent/evaluation/`): Benchmarking against baselines (random+filter, single-shot LLM) with metrics for design quality and reliability estimation.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
python scripts/run_agent.py \
  --target "antimicrobial peptide targeting gram-negative bacteria" \
  --properties hemolysis:low solubility:high toxicity:low \
  --output results/amp_design.json
```

Ablation (without reliability):
```bash
python scripts/run_agent.py \
  --target "antimicrobial peptide targeting gram-negative bacteria" \
  --properties hemolysis:low solubility:high \
  --no-reliability \
  --output results/amp_no_reliability.json
```

## Project Structure

```
peptagent/
├── agent/          # LLM orchestration loop, prompts, memory
├── tools/          # Peptide tool wrappers (ESM-2, properties, docking, etc.)
├── reliability/    # Ensemble, OOD detection, calibration, aggregation
├── evaluation/     # Metrics, baselines, benchmark runner
├── data/           # Dataset loaders and train/cal/test splitting
└── utils/          # Config, chemistry utilities
```

## Testing

```bash
pytest tests/
```
