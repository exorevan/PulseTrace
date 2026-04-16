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
    """LIME explanations for tabular and time series data."""

    def __init__(self, config: ExplainerConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Internal helpers — tabular
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
    # Internal helpers — time series
    # ------------------------------------------------------------------

    def _explain_ts_instance(
        self,
        model: ModelAdapter,
        instance_1d: np.ndarray,
        background: np.ndarray,
        n_segments: int,
    ) -> list[tuple[str, float]]:
        """Segment-masked LIME for a single time series instance.

        Divides the series into n_segments contiguous windows. Each perturbation
        keeps or replaces each segment with the per-timestep background mean.
        Returns (segment_name, weight) pairs.
        """
        T = len(instance_1d)
        seg_starts = [i * T // n_segments for i in range(n_segments)]
        seg_ends = [(i + 1) * T // n_segments for i in range(n_segments)]
        seg_ends[-1] = T

        segment_names = [
            f"seg_{i} [t{seg_starts[i]}-t{seg_ends[i] - 1}]"
            for i in range(n_segments)
        ]
        bg_mean = background.mean(axis=0)  # (T,)

        predict_fn = model.predict_proba if model.task == "classification" else model.predict

        def wrapped_predict(binary_matrix: np.ndarray) -> np.ndarray:
            out = []
            for row in binary_matrix:
                ts = instance_1d.copy()
                for seg_idx, keep in enumerate(row):
                    if not keep:
                        s, e = seg_starts[seg_idx], seg_ends[seg_idx]
                        ts[s:e] = bg_mean[s:e]
                result = predict_fn(ts.reshape(1, -1))
                out.append(result[0])
            return np.array(out)

        # Training data for LIME: random binary matrix gives the segment distribution
        rng = np.random.RandomState(42)
        n_train = max(200, self.config.num_samples)
        train_binary = rng.randint(0, 2, size=(n_train, n_segments)).astype(float)

        mode = "classification" if model.task == "classification" else "regression"
        lime_exp = LimeTabularExplainer(
            train_binary,
            feature_names=segment_names,
            categorical_features=list(range(n_segments)),
            mode=mode,
            discretize_continuous=False,
        )

        instance_binary = np.ones(n_segments)
        exp = lime_exp.explain_instance(
            instance_binary,
            wrapped_predict,
            num_features=self.config.num_features,
            num_samples=self.config.num_samples,
        )
        return exp.as_list()

    def _explain_ts_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        n_samples = min(self.config.global_samples, len(dataset))
        n_segments = self.config.n_segments
        background = dataset.X

        if model.task == "classification":
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            totals: dict[ty.Any, dict[str, float]] = {c: {} for c in classes}
            counts: dict[ty.Any, dict[str, int]] = {c: {} for c in classes}

            for i in range(n_samples):
                pairs = self._explain_ts_instance(model, dataset.instance(i), background, n_segments)
                cls = classes[0]
                for feat, weight in pairs:
                    totals[cls][feat] = totals[cls].get(feat, 0.0) + weight
                    counts[cls][feat] = counts[cls].get(feat, 0) + 1

            contributions = [
                FeatureContribution(feature=feat, weight=totals[cls][feat] / counts[cls][feat], label=cls)
                for cls in classes
                for feat in totals[cls]
            ]
            return ExplanationResult(
                mode="global", method="lime", task=model.task,
                target_name=dataset.target_name, contributions=contributions,
                base_values=None, global_samples=n_samples,
            )
        else:
            bg_for_base = dataset.X[np.random.choice(len(dataset), size=min(100, len(dataset)), replace=False)]
            reg_base: dict[str | int | None, float] = {None: float(model.predict(bg_for_base).mean())}  # type: ignore[dict-item]
            totals_r: dict[str, float] = {}
            counts_r: dict[str, int] = {}

            for i in range(n_samples):
                pairs = self._explain_ts_instance(model, dataset.instance(i), background, n_segments)
                for feat, weight in pairs:
                    totals_r[feat] = totals_r.get(feat, 0.0) + weight
                    counts_r[feat] = counts_r.get(feat, 0) + 1

            contributions = [
                FeatureContribution(feature=feat, weight=totals_r[feat] / counts_r[feat], label=None)
                for feat in totals_r
            ]
            return ExplanationResult(
                mode="global", method="lime", task=model.task,
                target_name=dataset.target_name, contributions=contributions,
                base_values=reg_base, global_samples=n_samples,
            )

    def _explain_ts_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        n_segments = self.config.n_segments
        x = instance.instance(0)
        pairs = self._explain_ts_instance(model, x, dataset.X, n_segments)

        if model.task == "classification":
            proba = model.predict_proba(x.reshape(1, -1))[0]
            predicted_idx = int(np.argmax(proba))
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            predicted_label = classes[predicted_idx]
            contributions = [
                FeatureContribution(feature=feat, weight=weight, label=predicted_label)
                for feat, weight in pairs
            ]
            base_values: dict[ty.Any, float] = {predicted_label: float(proba[predicted_idx])}
        else:
            predicted_value = float(model.predict(x.reshape(1, -1))[0])
            contributions = [
                FeatureContribution(feature=feat, weight=weight, label=None)
                for feat, weight in pairs
            ]
            base_values = {None: predicted_value}

        return ExplanationResult(
            mode="local", method="lime", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values,
        )

    # ------------------------------------------------------------------
    # Internal helpers — image
    # ------------------------------------------------------------------

    def _explain_image_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        from lime.lime_image import LimeImageExplainer as _LimeImage
        n_samples = min(self.config.global_samples, len(dataset))
        lime_img = _LimeImage()

        if model.task == "classification":
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            n_classes = len(classes)
            predict_fn = model.predict_proba

            totals: dict[ty.Any, dict[str, float]] = {c: {} for c in classes}
            counts: dict[ty.Any, dict[str, int]] = {c: {} for c in classes}

            for i in range(n_samples):
                image = dataset.instance(i)  # (H, W, C)
                exp = lime_img.explain_instance(
                    image.astype(np.double),
                    predict_fn,
                    num_features=self.config.num_features,
                    num_samples=self.config.num_samples,
                    labels=list(range(n_classes)),
                )
                for class_idx, cls in enumerate(classes):
                    for seg_id, weight in exp.local_exp.get(class_idx, []):
                        feat = f"region_{seg_id}"
                        totals[cls][feat] = totals[cls].get(feat, 0.0) + weight
                        counts[cls][feat] = counts[cls].get(feat, 0) + 1

            contributions = [
                FeatureContribution(feature=feat, weight=totals[cls][feat] / counts[cls][feat], label=cls)
                for cls in classes
                for feat in totals[cls]
            ]
            return ExplanationResult(
                mode="global", method="lime", task=model.task,
                target_name=dataset.target_name, contributions=contributions,
                base_values=None, global_samples=n_samples,
            )
        else:
            def predict_fn_reg(images: np.ndarray) -> np.ndarray:
                return model.predict(images).reshape(-1, 1)

            bg_sample = dataset.X[np.random.choice(len(dataset), size=min(100, len(dataset)), replace=False)]
            reg_base: dict[ty.Any, float] = {None: float(model.predict(bg_sample).mean())}  # type: ignore[dict-item]
            totals_r: dict[str, float] = {}
            counts_r: dict[str, int] = {}

            for i in range(n_samples):
                image = dataset.instance(i)  # (H, W, C)
                exp = lime_img.explain_instance(
                    image.astype(np.double),
                    predict_fn_reg,
                    num_features=self.config.num_features,
                    num_samples=self.config.num_samples,
                    labels=[0],
                )
                for seg_id, weight in exp.local_exp.get(0, []):
                    feat = f"region_{seg_id}"
                    totals_r[feat] = totals_r.get(feat, 0.0) + weight
                    counts_r[feat] = counts_r.get(feat, 0) + 1

            contributions = [
                FeatureContribution(feature=feat, weight=totals_r[feat] / counts_r[feat], label=None)
                for feat in totals_r
            ]
            return ExplanationResult(
                mode="global", method="lime", task=model.task,
                target_name=dataset.target_name, contributions=contributions,
                base_values=reg_base, global_samples=n_samples,
            )

    def _explain_image_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        from lime.lime_image import LimeImageExplainer as _LimeImage
        lime_img = _LimeImage()
        image = instance.instance(0)  # (H, W, C)

        if model.task == "classification":
            predict_fn = model.predict_proba
            proba = predict_fn(image.reshape(1, *image.shape))[0]
            predicted_idx = int(np.argmax(proba))
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            predicted_label = classes[predicted_idx]
            n_classes = len(classes)

            exp = lime_img.explain_instance(
                image.astype(np.double),
                predict_fn,
                num_features=self.config.num_features,
                num_samples=self.config.num_samples,
                labels=list(range(n_classes)),
            )
            contributions = [
                FeatureContribution(feature=f"region_{seg_id}", weight=weight, label=predicted_label)
                for seg_id, weight in exp.local_exp.get(predicted_idx, [])
            ]
            base_values: dict[ty.Any, float] = {predicted_label: float(proba[predicted_idx])}
        else:
            def predict_fn_reg(images: np.ndarray) -> np.ndarray:
                return model.predict(images).reshape(-1, 1)

            predicted_value = float(model.predict(image.reshape(1, *image.shape))[0])
            exp = lime_img.explain_instance(
                image.astype(np.double),
                predict_fn_reg,
                num_features=self.config.num_features,
                num_samples=self.config.num_samples,
                labels=[0],
            )
            contributions = [
                FeatureContribution(feature=f"region_{seg_id}", weight=weight, label=None)
                for seg_id, weight in exp.local_exp.get(0, [])
            ]
            base_values = {None: predicted_value}

        return ExplanationResult(
            mode="local", method="lime", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        from pulsetrace.data.image_dataset import ImageDataset
        from pulsetrace.data.timeseries_dataset import TimeSeriesDataset
        if isinstance(dataset, ImageDataset):
            return self._explain_image_global(model, dataset)
        if isinstance(dataset, TimeSeriesDataset):
            return self._explain_ts_global(model, dataset)
        return self._explain_tabular_global(model, dataset)

    def _explain_tabular_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        mode = "classification" if model.task == "classification" else "regression"
        lime_exp = self._make_tabular_explainer(dataset, mode)
        predict_fn = model.predict_proba if model.task == "classification" else model.predict

        n_samples = min(self.config.global_samples, len(dataset))

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

            background = dataset.X[np.random.choice(len(dataset), size=min(100, len(dataset)), replace=False)]
            reg_base: dict[str | int | None, float] = {None: float(model.predict(background).mean())}  # type: ignore[dict-item]

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
                base_values=reg_base,
                global_samples=n_samples,
            )

        # Classification global
        return ExplanationResult(
            mode="global",
            method="lime",
            task=model.task,
            target_name=dataset.target_name,
            contributions=contributions,
            base_values=None,
            global_samples=n_samples,
        )

    def explain_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        from pulsetrace.data.image_dataset import ImageDataset
        from pulsetrace.data.timeseries_dataset import TimeSeriesDataset
        if isinstance(dataset, ImageDataset):
            return self._explain_image_local(model, instance, dataset)
        if isinstance(dataset, TimeSeriesDataset):
            return self._explain_ts_local(model, instance, dataset)
        return self._explain_tabular_local(model, instance, dataset)

    def _explain_tabular_local(
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
            base_values: dict[ty.Any, float] = {
                predicted_label: float(proba[predicted_idx])
            }
        else:
            predicted_value = float(model.predict(x.reshape(1, -1))[0])
            contributions = [
                FeatureContribution(feature=feat, weight=weight, label=None)
                for feat, weight in exp.as_list()
            ]
            base_values = {None: predicted_value}

        return ExplanationResult(
            mode="local",
            method="lime",
            task=model.task,
            target_name=dataset.target_name,
            contributions=contributions,
            base_values=base_values,
        )
