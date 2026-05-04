from __future__ import annotations

import argparse
from pathlib import Path

from pulsetrace.adapters import build_adapter
from pulsetrace.config import load_config
from pulsetrace.data import load_dataset
from pulsetrace.explainers import build_explainer
from pulsetrace.output import render
from pulsetrace.utils.logger import setup_logging


def run(cfg_path: Path) -> None:
    config = load_config(cfg_path)
    setup_logging(config.logging)

    adapter = build_adapter(config.model)
    dataset = load_dataset(config.dataset)

    explainer = build_explainer(config.explainer)

    if config.app.mode == "global":
        result = explainer.explain_global(adapter, dataset)
    else:
        if config.local is None:
            raise ValueError(
                "local.dataset must be configured in the YAML when app.mode is 'local'."
            )
        instance = load_dataset(config.local.dataset)
        result = explainer.explain_local(adapter, instance, dataset)

    render(result, output_format=config.app.output_format)


def _resolve_cfg(value: str) -> Path:
    """Accept a full path or a bare config name (e.g. 'ts_italy_lime').

    Bare names are resolved relative to the 'configs/' directory, with or
    without a '.yaml' extension.
    """
    p = Path(value)
    if p.suffix in {".yaml", ".yml"} or p.exists():
        return p
    candidates = [
        Path("configs") / f"{value}.yaml",
        Path("configs") / value,
    ]
    for c in candidates:
        if c.exists():
            return c
    return p  # fall through to let load_config raise a clear error


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="PulseTrace — Explainable AI model interpreter"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Path to the YAML config file or a bare config name (e.g. ts_italy_lime)",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Bare config name as a positional argument (e.g. ts_italy_lime)",
    )
    args = parser.parse_args()
    raw = args.cfg or args.config
    if raw is None:
        parser.error("provide a config via --cfg or as a positional argument")
    run(_resolve_cfg(raw))
