"""Configuration loading with OmegaConf."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def load_config(
    path: str | None = None,
    overrides: list[str] | None = None,
) -> OmegaConf:
    """Load configuration from YAML files with optional overrides.

    Args:
        path: Path to experiment config (merged on top of default).
        overrides: List of dotlist overrides (e.g., ["llm.model=gpt-4o-mini"]).

    Returns:
        Merged OmegaConf configuration.
    """
    base_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    cfg = OmegaConf.load(str(base_path))

    if path:
        exp_cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(cfg, exp_cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg
