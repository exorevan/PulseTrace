from __future__ import annotations

from pathlib import Path

import yaml

from .schema import PulseTraceConfig


def load_config(path: Path) -> PulseTraceConfig:
    """Load and validate a PulseTrace YAML config file.

    Raises:
        FileNotFoundError: if the YAML file does not exist.
        ValidationError: if the config is structurally invalid.
        ValueError: if the file is not valid YAML.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")

    return PulseTraceConfig.model_validate(raw)


__all__ = ["load_config", "PulseTraceConfig"]
