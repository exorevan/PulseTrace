from __future__ import annotations

from pulsetrace.config.schema import ExplainerConfig

from .base import BaseExplainer
from .lime import LimeExplainer
from .result import ExplanationResult, FeatureContribution
from .shap import ShapExplainer


def build_explainer(config: ExplainerConfig) -> BaseExplainer:
    if config.type == "lime":
        return LimeExplainer(config)
    if config.type == "shap":
        return ShapExplainer(config)
    raise ValueError(f"Unknown explainer type: '{config.type}'")


__all__ = [
    "BaseExplainer",
    "LimeExplainer",
    "ShapExplainer",
    "ExplanationResult",
    "FeatureContribution",
    "build_explainer",
]
