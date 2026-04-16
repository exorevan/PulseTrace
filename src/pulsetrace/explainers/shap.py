from __future__ import annotations

import numpy as np
import shap

from pulsetrace.adapters.base import ModelAdapter
from pulsetrace.config.schema import ExplainerConfig
from pulsetrace.data.dataset import Dataset

from .base import BaseExplainer
from .result import ExplanationResult, FeatureContribution


class ShapExplainer(BaseExplainer):
    """SHAP explanations for tabular and time series data."""

    def __init__(self, config: ExplainerConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Internal helpers — tabular
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
        weights = np.asarray(weights).ravel()  # ensure 1D
        top_indices = np.argsort(np.abs(weights))[::-1][: self.config.num_features]
        return [
            FeatureContribution(
                feature=feature_names[int(i)],
                weight=float(weights[int(i)]),
                label=label,
            )
            for i in top_indices
        ]

    # ------------------------------------------------------------------
    # Internal helpers — time series
    # ------------------------------------------------------------------

    def _make_kernel_explainer(
        self,
        model: ModelAdapter,
        background: np.ndarray,
        feature_names: list[str],
    ) -> shap.KernelExplainer:
        predict_fn = (
            model.predict_proba if model.task == "classification" else model.predict
        )
        summary = shap.kmeans(background, min(10, len(background)))
        return shap.KernelExplainer(predict_fn, summary)

    def _explain_ts_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        background = dataset.X[:100] if len(dataset) > 100 else dataset.X
        explainer = self._make_kernel_explainer(model, background, dataset.feature_names)

        sample = dataset.X[:10] if len(dataset) > 10 else dataset.X
        shap_values = explainer.shap_values(sample)

        contributions: list[FeatureContribution] = []
        base_values: dict[str | int | None, float] = {}

        if model.task == "classification":
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            sv_arr = np.array(shap_values)  # works for both list-of-arrays and ndarray
            # sv_arr shape: (n_classes, n_samples, n_features) if list, or (n_samples, n_features)
            if sv_arr.ndim == 3:
                for class_idx, cls in enumerate(classes):
                    avg_abs = np.mean(np.abs(sv_arr[class_idx]), axis=0)
                    contributions.extend(
                        self._top_contributions(avg_abs, dataset.feature_names, label=cls)
                    )
            else:
                avg_abs = np.mean(np.abs(sv_arr), axis=0)
                contributions.extend(
                    self._top_contributions(avg_abs, dataset.feature_names, label=None)
                )
            ev = explainer.expected_value
            base_values[None] = float(ev[0]) if hasattr(ev, "__len__") else float(ev)
        else:
            avg_abs = np.mean(np.abs(shap_values), axis=0)
            contributions.extend(
                self._top_contributions(avg_abs, dataset.feature_names, label=None)
            )
            ev = explainer.expected_value
            base_values[None] = float(ev) if np.ndim(ev) == 0 else float(ev[0])

        return ExplanationResult(
            mode="global", method="shap", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values,
        )

    def _explain_ts_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        background = dataset.X[:100] if len(dataset) > 100 else dataset.X
        explainer = self._make_kernel_explainer(model, background, dataset.feature_names)

        x = instance.instance(0).reshape(1, -1)
        shap_values = explainer.shap_values(x)

        contributions: list[FeatureContribution] = []
        base_values: dict[str | int | None, float] = {}

        if model.task == "classification":
            proba = model.predict_proba(x)[0]
            predicted_idx = int(np.argmax(proba))
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            predicted_label = classes[predicted_idx]
            sv_arr = np.array(shap_values)  # (n_classes, 1, n_features) or (1, n_features)
            if sv_arr.ndim == 3:
                vals = sv_arr[predicted_idx][0]  # (n_features,)
            else:
                vals = sv_arr[0]
            contributions.extend(
                self._top_contributions(vals, dataset.feature_names, label=predicted_label)
            )
            base_values[predicted_label] = float(proba[predicted_idx])
        else:
            sv_arr = np.array(shap_values)
            vals = sv_arr[0] if sv_arr.ndim >= 2 else sv_arr
            contributions.extend(
                self._top_contributions(vals, dataset.feature_names, label=None)
            )
            ev = explainer.expected_value
            base_values[None] = float(ev) if np.ndim(ev) == 0 else float(ev[0])

        return ExplanationResult(
            mode="local", method="shap", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values,
        )

    # ------------------------------------------------------------------
    # Internal helpers — image
    # ------------------------------------------------------------------

    def _explain_image_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        import skimage.segmentation

        sample = dataset.X[:min(10, len(dataset))]   # (n, H, W, C)
        image = sample[0]                             # reference image for segmentation
        bg_mean = sample.mean(axis=0)                 # (H, W, C) — fill value for masker

        masker = shap.maskers.Image(bg_mean, bg_mean.shape)
        predict_fn = model.predict_proba if model.task == "classification" else model.predict
        explainer = shap.Explainer(predict_fn, masker)

        shap_values = explainer(sample)
        vals = shap_values.values
        # classification (predict_proba → (n, k)): vals.shape == (n, H, W, C, k)
        # regression     (predict → (n,)):         vals.shape == (n, H, W, C)

        segments = skimage.segmentation.quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

        contributions: list[FeatureContribution] = []
        base_values: dict[str | int | None, float] = {}

        if model.task == "classification":
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            for class_idx, cls in enumerate(classes):
                class_vals = vals[..., class_idx]                         # (n, H, W, C)
                avg_abs = np.mean(np.abs(class_vals), axis=0)             # (H, W, C)
                avg_abs_hw = avg_abs.mean(axis=-1)                        # (H, W)
                region_weights: dict[int, float] = {
                    int(sid): float(avg_abs_hw[segments == sid].mean())
                    for sid in np.unique(segments)
                }
                top = sorted(region_weights.items(), key=lambda x: x[1], reverse=True)[: self.config.num_features]
                for seg_id, w in top:
                    contributions.append(FeatureContribution(feature=f"region_{seg_id}", weight=w, label=cls))
            ev = shap_values.base_values
            base_values[None] = float(np.mean(ev)) if ev is not None else 0.0
        else:
            avg_abs = np.mean(np.abs(vals), axis=0)   # (H, W, C)
            avg_abs_hw = avg_abs.mean(axis=-1)         # (H, W)
            region_weights_r: dict[int, float] = {
                int(sid): float(avg_abs_hw[segments == sid].mean())
                for sid in np.unique(segments)
            }
            top_r = sorted(region_weights_r.items(), key=lambda x: x[1], reverse=True)[: self.config.num_features]
            for seg_id, w in top_r:
                contributions.append(FeatureContribution(feature=f"region_{seg_id}", weight=w, label=None))
            ev = shap_values.base_values
            base_values[None] = float(np.mean(ev)) if ev is not None else 0.0

        return ExplanationResult(
            mode="global", method="shap", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values,
        )

    def _explain_image_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        import skimage.segmentation

        image = instance.instance(0)                                      # (H, W, C)
        bg_mean = dataset.X[:min(10, len(dataset))].mean(axis=0)          # (H, W, C)

        masker = shap.maskers.Image(bg_mean, bg_mean.shape)
        predict_fn = model.predict_proba if model.task == "classification" else model.predict
        explainer = shap.Explainer(predict_fn, masker)

        shap_values = explainer(image.reshape(1, *image.shape))
        vals = shap_values.values   # (1, H, W, C) regression  OR  (1, H, W, C, n_classes)

        segments = skimage.segmentation.quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

        contributions: list[FeatureContribution] = []
        base_values: dict[str | int | None, float] = {}

        if model.task == "classification":
            proba = model.predict_proba(image.reshape(1, *image.shape))[0]
            predicted_idx = int(np.argmax(proba))
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            predicted_label = classes[predicted_idx]

            class_vals = vals[0, ..., predicted_idx]       # (H, W, C)
            avg_abs_hw = np.abs(class_vals).mean(axis=-1)  # (H, W)
            region_weights: dict[int, float] = {
                int(sid): float(avg_abs_hw[segments == sid].mean())
                for sid in np.unique(segments)
            }
            top = sorted(region_weights.items(), key=lambda x: x[1], reverse=True)[: self.config.num_features]
            for seg_id, w in top:
                contributions.append(FeatureContribution(feature=f"region_{seg_id}", weight=w, label=predicted_label))
            base_values[predicted_label] = float(proba[predicted_idx])
        else:
            avg_abs_hw = np.abs(vals[0]).mean(axis=-1)     # (H, W)
            region_weights_r: dict[int, float] = {
                int(sid): float(avg_abs_hw[segments == sid].mean())
                for sid in np.unique(segments)
            }
            top_r = sorted(region_weights_r.items(), key=lambda x: x[1], reverse=True)[: self.config.num_features]
            for seg_id, w in top_r:
                contributions.append(FeatureContribution(feature=f"region_{seg_id}", weight=w, label=None))
            ev = shap_values.base_values
            base_values[None] = float(np.mean(ev)) if ev is not None else 0.0

        return ExplanationResult(
            mode="local", method="shap", task=model.task,
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
        background = dataset.X[:100] if len(dataset) > 100 else dataset.X
        explainer = self._make_shap_explainer(model, background, dataset.feature_names)

        sample = dataset.X[:10]
        shap_values = explainer(sample)
        values: np.ndarray = shap_values.values  # (n_samples, n_features[, n_classes])
        raw_base: np.ndarray = shap_values.base_values  # (n_samples,) or (n_samples, n_classes)

        contributions: list[FeatureContribution] = []
        base_values: dict[str | int | None, float] = {}

        if model.task == "classification":
            if values.ndim == 3:
                classes = dataset.classes if dataset.classes is not None else np.array([0])
                for class_idx, cls in enumerate(classes):
                    avg_abs = np.mean(np.abs(values[:, :, class_idx]), axis=0)
                    contributions.extend(
                        self._top_contributions(avg_abs, dataset.feature_names, label=cls)
                    )
                    if raw_base.ndim == 2:
                        base_values[cls] = float(np.mean(raw_base[:, class_idx]))
                    else:
                        base_values[cls] = float(np.mean(raw_base))
            else:
                avg_abs = np.mean(np.abs(values), axis=0)
                contributions.extend(
                    self._top_contributions(avg_abs, dataset.feature_names, label=None)
                )
                base_values[None] = float(np.mean(raw_base))
        else:
            avg_abs = np.mean(np.abs(values), axis=0)
            contributions.extend(
                self._top_contributions(avg_abs, dataset.feature_names, label=None)
            )
            base_values[None] = float(np.mean(raw_base))

        return ExplanationResult(
            mode="global",
            method="shap",
            task=model.task,
            target_name=dataset.target_name,
            contributions=contributions,
            base_values=base_values,
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
        background = dataset.X[:100] if len(dataset) > 100 else dataset.X
        explainer = self._make_shap_explainer(model, background, dataset.feature_names)

        x = instance.instance(0).reshape(1, -1)
        shap_values = explainer(x)
        values: np.ndarray = shap_values.values[0]  # (n_features[, n_classes])
        raw_base: np.ndarray = shap_values.base_values[0]  # scalar or (n_classes,)

        contributions: list[FeatureContribution] = []
        base_values: dict[str | int | None, float] = {}

        if model.task == "classification":
            if values.ndim == 2:
                proba = model.predict_proba(x)[0]
                predicted_idx = int(np.argmax(proba))
                classes = dataset.classes if dataset.classes is not None else np.array([0])
                predicted_label = classes[predicted_idx]
                contributions.extend(
                    self._top_contributions(
                        values[:, predicted_idx], dataset.feature_names, label=predicted_label
                    )
                )
                base_values[predicted_label] = (
                    float(raw_base[predicted_idx])
                    if np.ndim(raw_base) > 0
                    else float(raw_base)
                )
            else:
                contributions.extend(
                    self._top_contributions(values, dataset.feature_names, label=None)
                )
                base_values[None] = float(np.mean(raw_base))
        else:
            contributions.extend(
                self._top_contributions(values, dataset.feature_names, label=None)
            )
            base_values[None] = float(raw_base) if np.ndim(raw_base) == 0 else float(raw_base[0])

        return ExplanationResult(
            mode="local",
            method="shap",
            task=model.task,
            target_name=dataset.target_name,
            contributions=contributions,
            base_values=base_values,
        )
