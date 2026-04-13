from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FeatureContribution:
    """Weight of one feature toward a prediction.

    label is the class name for classification results, None for regression.
    """

    feature: str
    weight: float
    label: str | int | None = None


@dataclass(frozen=True)
class ExplanationResult:
    """Structured output of an explanation run.

    contributions contains one FeatureContribution per (feature, label) pair
    for classification global results; one per feature (label=None) for
    regression and for local results.
    """

    mode: Literal["global", "local"]
    method: Literal["lime", "shap"]
    task: Literal["classification", "regression"]
    target_name: str
    contributions: list[FeatureContribution]
