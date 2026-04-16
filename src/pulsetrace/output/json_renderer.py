from __future__ import annotations

import json

from pulsetrace.explainers.result import ExplanationResult


def render(result: ExplanationResult) -> None:
    """Print an ExplanationResult as JSON to stdout."""
    data: dict = {
        "method": result.method,
        "mode": result.mode,
        "task": result.task,
        "target_name": result.target_name,
        "contributions": [
            {
                "feature": c.feature,
                "weight": c.weight,
                "label": c.label,
            }
            for c in result.contributions
        ],
    }
    if result.base_values is not None:
        data["base_values"] = {
            (str(k) if k is not None else "null"): v
            for k, v in result.base_values.items()
        }
    if result.global_samples is not None:
        data["global_samples"] = result.global_samples

    print(json.dumps(data, indent=2))
