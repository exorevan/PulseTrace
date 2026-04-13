from __future__ import annotations

import numpy as np
import shap

from pulsetrace.adapters.base import ModelAdapter
from pulsetrace.config.schema import ExplainerConfig
from pulsetrace.data.dataset import Dataset

from .base import BaseExplainer
from .result import ExplanationResult, FeatureContribution


class ShapExplainer(BaseExplainer):
    """SHAP explanations for tabular data."""

    def __init__(self, config: ExplainerConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_shap_explainer(
        self,
        model: ModelAdapter,
        background: np.ndarray,
        feature_names: list[str],
    ) -> shap.Explainer:
        predict_fn = (
            model.predict_proba if model.task == "classification" else model.predict
        )
        return shap.Explainer(predict_fn, background, feature_names=feature_names)

    def _top_contributions(
        self,
        weights: np.ndarray,
        feature_names: list[str],
        label: str | int | None,
    ) -> list[FeatureContribution]:
        """Return the top-N contributions sorted by absolute weight."""
        top_indices = np.argsort(np.abs(weights))[::-1][: self.config.num_features]
        return [
            FeatureContribution(
                feature=feature_names[i],
                weight=float(weights[i]),
                label=label,
            )
            for i in top_indices
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        background = dataset.X[:100] if len(dataset) > 100 else dataset.X
        explainer = self._make_shap_explainer(model, background, dataset.feature_names)

        sample = dataset.X[:10]
        shap_values = explainer(sample)
        values: np.ndarray = shap_values.values  # (n_samples, n_features[, n_classes])

        contributions: list[FeatureContribution] = []

        if model.task == "classification":
            if values.ndim == 3:
                # Standard case: (n_samples, n_features, n_classes)
                # One set of contributions per class, labelled by class value.
                classes = dataset.classes if dataset.classes is not None else np.array([0])
                for class_idx, cls in enumerate(classes):
                    avg_abs = np.mean(np.abs(values[:, :, class_idx]), axis=0)
                    contributions.extend(
                        self._top_contributions(avg_abs, dataset.feature_names, label=cls)
                    )
            else:
                # 2D fallback: (n_samples, n_features)
                # SHAP collapsed the class dimension (e.g. KernelExplainer on binary
                # classifiers, or a squeezed LinearExplainer output). Per-class breakdown
                # is unavailable — report overall mean-absolute importance, label=None.
                avg_abs = np.mean(np.abs(values), axis=0)
                contributions.extend(
                    self._top_contributions(avg_abs, dataset.feature_names, label=None)
                )
        else:
            # Regression: (n_samples, n_features)
            avg_abs = np.mean(np.abs(values), axis=0)
            contributions.extend(
                self._top_contributions(avg_abs, dataset.feature_names, label=None)
            )

        return ExplanationResult(
            mode="global",
            method="shap",
            task=model.task,
            target_name=dataset.target_name,
            contributions=contributions,
        )

    def explain_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        background = dataset.X[:100] if len(dataset) > 100 else dataset.X
        explainer = self._make_shap_explainer(model, background, dataset.feature_names)

        x = instance.instance(0).reshape(1, -1)
        shap_values = explainer(x)
        values: np.ndarray = shap_values.values[0]  # (n_features[, n_classes])

        contributions: list[FeatureContribution] = []

        if model.task == "classification":
            if values.ndim == 2:
                # Standard case: (n_features, n_classes)
                # Show contributions for the predicted class only.
                proba = model.predict_proba(x)[0]
                predicted_idx = int(np.argmax(proba))
                classes = dataset.classes if dataset.classes is not None else np.array([0])
                predicted_label = classes[predicted_idx]
                contributions.extend(
                    self._top_contributions(
                        values[:, predicted_idx], dataset.feature_names, label=predicted_label
                    )
                )
            else:
                # 1D fallback: (n_features,)
                # SHAP collapsed the class dimension. Per-class breakdown is unavailable —
                # report feature-level contributions, label=None.
                contributions.extend(
                    self._top_contributions(values, dataset.feature_names, label=None)
                )
        else:
            # Regression: (n_features,)
            contributions.extend(
                self._top_contributions(values, dataset.feature_names, label=None)
            )

        return ExplanationResult(
            mode="local",
            method="shap",
            task=model.task,
            target_name=dataset.target_name,
            contributions=contributions,
        )
