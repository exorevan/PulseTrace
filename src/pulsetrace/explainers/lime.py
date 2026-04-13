from __future__ import annotations

import typing as ty

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from pulsetrace.adapters.base import ModelAdapter
from pulsetrace.config.schema import ExplainerConfig
from pulsetrace.data.dataset import Dataset

from .base import BaseExplainer
from .result import ExplanationResult, FeatureContribution


class LimeExplainer(BaseExplainer):
    """LIME explanations for tabular data."""

    def __init__(self, config: ExplainerConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_tabular_explainer(
        self, dataset: Dataset, mode: str
    ) -> LimeTabularExplainer:
        return LimeTabularExplainer(
            dataset.X,
            feature_names=dataset.feature_names,
            mode=mode,
            discretize_continuous=True,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        mode = "classification" if model.task == "classification" else "regression"
        lime_exp = self._make_tabular_explainer(dataset, mode)
        predict_fn = model.predict_proba if model.task == "classification" else model.predict

        n_samples = min(10, len(dataset))

        if model.task == "classification":
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            n_classes = len(classes)

            totals: dict[ty.Any, dict[str, float]] = {c: {} for c in classes}
            counts: dict[ty.Any, dict[str, int]] = {c: {} for c in classes}

            for i in range(n_samples):
                exp = lime_exp.explain_instance(
                    dataset.instance(i),
                    predict_fn,
                    num_features=self.config.num_features,
                    labels=list(range(n_classes)),
                )
                for idx, cls in enumerate(classes):
                    for feat, weight in exp.as_list(label=idx):
                        totals[cls][feat] = totals[cls].get(feat, 0.0) + weight
                        counts[cls][feat] = counts[cls].get(feat, 0) + 1

            contributions = [
                FeatureContribution(
                    feature=feat,
                    weight=totals[cls][feat] / counts[cls][feat],
                    label=cls,
                )
                for cls in classes
                for feat in totals[cls]
            ]

        else:
            totals_r: dict[str, float] = {}
            counts_r: dict[str, int] = {}

            for i in range(n_samples):
                exp = lime_exp.explain_instance(
                    dataset.instance(i),
                    predict_fn,
                    num_features=self.config.num_features,
                )
                for feat, weight in exp.as_list():
                    totals_r[feat] = totals_r.get(feat, 0.0) + weight
                    counts_r[feat] = counts_r.get(feat, 0) + 1

            contributions = [
                FeatureContribution(
                    feature=feat,
                    weight=totals_r[feat] / counts_r[feat],
                    label=None,
                )
                for feat in totals_r
            ]

        return ExplanationResult(
            mode="global",
            method="lime",
            task=model.task,
            target_name=dataset.target_name,
            contributions=contributions,
        )

    def explain_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        mode = "classification" if model.task == "classification" else "regression"
        lime_exp = self._make_tabular_explainer(dataset, mode)
        predict_fn = model.predict_proba if model.task == "classification" else model.predict

        x = instance.instance(0)
        exp = lime_exp.explain_instance(
            x, predict_fn, num_features=self.config.num_features
        )

        if model.task == "classification":
            proba = model.predict_proba(x.reshape(1, -1))[0]
            predicted_idx = int(np.argmax(proba))
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            predicted_label = classes[predicted_idx]

            contributions = [
                FeatureContribution(feature=feat, weight=weight, label=predicted_label)
                for feat, weight in exp.as_list()
            ]
        else:
            contributions = [
                FeatureContribution(feature=feat, weight=weight, label=None)
                for feat, weight in exp.as_list()
            ]

        return ExplanationResult(
            mode="local",
            method="lime",
            task=model.task,
            target_name=dataset.target_name,
            contributions=contributions,
        )
