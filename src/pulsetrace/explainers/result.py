from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ImagePanel:
    """Original image alongside its XAI explanation overlay, encoded as base64 PNGs."""

    title: str
    original_b64: str
    explanation_b64: str
    confidence: float | None = None


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

    base_values maps each class label (or None for regression) to the baseline
    model output before feature contributions are applied (LIME intercept /
    SHAP expected value).
    """

    mode: Literal["global", "local"]
    method: Literal["lime", "shap", "ig"]
    task: Literal["classification", "regression"]
    target_name: str
    contributions: list[FeatureContribution]
    base_values: dict[str | int | None, float] | None = None
    global_samples: int | None = None
    image_panels: list[ImagePanel] | None = None
