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


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="PulseTrace — Explainable AI model interpreter"
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        required=True,
        help="Path to the YAML configuration file",
    )
    run(parser.parse_args().cfg)
