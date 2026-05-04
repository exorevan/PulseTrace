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
        from .ts_utils import expand_segment_weights, ts_panel

        n_samples = min(self.config.global_samples, len(dataset))
        n_segments = self.config.n_segments
        background = dataset.X
        T = background.shape[1]

        # Background base value (mean prediction on a small background slice)
        bg_slice = background[:min(20, len(background))]
        predict_fn = model.predict_proba if model.task == "classification" else model.predict
        panels = []

        if model.task == "classification":
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            totals: dict[ty.Any, dict[str, float]] = {c: {} for c in classes}
            counts: dict[ty.Any, dict[str, int]] = {c: {} for c in classes}
            bg_proba = predict_fn(bg_slice)   # (n_bg, n_classes)
            base_values_cls: dict[str | int | None, float] = {
                int(cls): float(np.mean(bg_proba[:, ci]))
                for ci, cls in enumerate(classes)
            }

            for i in range(n_samples):
                series = dataset.instance(i)
                pairs = self._explain_ts_instance(model, series, background, n_segments)
                cls = classes[0]
                for feat, weight in pairs:
                    totals[cls][feat] = totals[cls].get(feat, 0.0) + weight
                    counts[cls][feat] = counts[cls].get(feat, 0) + 1

                proba = predict_fn(series.reshape(1, -1))[0]
                predicted_idx = int(np.argmax(proba))
                predicted_label = classes[predicted_idx]
                confidence = float(proba[predicted_idx])

                weights_ts = expand_segment_weights(pairs, T)
                panels.append(ts_panel(
                    series, weights_ts,
                    title=f"Sample {i} — pred: {predicted_label}",
                    confidence=confidence,
                ))

            contributions = [
                FeatureContribution(feature=feat, weight=totals[cls][feat] / counts[cls][feat], label=cls)
                for cls in classes
                for feat in totals[cls]
            ]
            return ExplanationResult(
                mode="global", method="lime", task=model.task,
                target_name=dataset.target_name, contributions=contributions,
                base_values=base_values_cls, global_samples=n_samples,
                image_panels=panels,
            )
        else:
            bg_for_base = dataset.X[np.random.choice(len(dataset), size=min(100, len(dataset)), replace=False)]
            reg_base: dict[str | int | None, float] = {None: float(model.predict(bg_for_base).mean())}  # type: ignore[dict-item]
            totals_r: dict[str, float] = {}
            counts_r: dict[str, int] = {}

            for i in range(n_samples):
                series = dataset.instance(i)
                pairs = self._explain_ts_instance(model, series, background, n_segments)
                for feat, weight in pairs:
                    totals_r[feat] = totals_r.get(feat, 0.0) + weight
                    counts_r[feat] = counts_r.get(feat, 0) + 1

                predicted_value = float(model.predict(series.reshape(1, -1))[0])
                weights_ts = expand_segment_weights(pairs, T)
                panels.append(ts_panel(
                    series, weights_ts,
                    title=f"Sample {i} — pred: {predicted_value:.4f}",
                ))

            contributions = [
                FeatureContribution(feature=feat, weight=totals_r[feat] / counts_r[feat], label=None)
                for feat in totals_r
            ]
            return ExplanationResult(
                mode="global", method="lime", task=model.task,
                target_name=dataset.target_name, contributions=contributions,
                base_values=reg_base, global_samples=n_samples,
                image_panels=panels,
            )

    def _explain_ts_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        from .ts_utils import expand_segment_weights, ts_panel

        n_segments = self.config.n_segments
        T = dataset.X.shape[1]
        x = instance.instance(0)
        pairs = self._explain_ts_instance(model, x, dataset.X, n_segments)
        weights_ts = expand_segment_weights(pairs, T)

        if model.task == "classification":
            proba = model.predict_proba(x.reshape(1, -1))[0]
            predicted_idx = int(np.argmax(proba))
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            predicted_label = classes[predicted_idx]
            confidence = float(proba[predicted_idx])
            bg_proba = model.predict_proba(dataset.X[:min(20, len(dataset))])
            contributions = [
                FeatureContribution(feature=feat, weight=weight, label=predicted_label)
                for feat, weight in pairs
            ]
            base_values: dict[ty.Any, float] = {
                int(cls): float(np.mean(bg_proba[:, ci]))
                for ci, cls in enumerate(classes)
            }
            panel = ts_panel(
                x, weights_ts,
                title=f"pred: {predicted_label}",
                confidence=confidence,
            )
        else:
            predicted_value = float(model.predict(x.reshape(1, -1))[0])
            base_val = float(model.predict(dataset.X[:min(20, len(dataset))]).mean())
            contributions = [
                FeatureContribution(feature=feat, weight=weight, label=None)
                for feat, weight in pairs
            ]
            base_values = {None: base_val}
            panel = ts_panel(
                x, weights_ts,
                title=f"pred: {predicted_value:.4f}",
            )

        return ExplanationResult(
            mode="local", method="lime", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values,
            image_panels=[panel],
        )

    # ------------------------------------------------------------------
    # Internal helpers — image
    # ------------------------------------------------------------------

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert (H, W, 1) grayscale to (H, W, 3) for LIME's quickshift segmenter."""
        if image.shape[-1] == 1:
            return np.repeat(image, 3, axis=-1)
        return image

    @staticmethod
    def _wrap_predict_rgb(predict_fn: ty.Callable, n_channels: int) -> ty.Callable:
        """If the model expects 1-channel input, collapse LIME's 3-channel perturbations."""
        if n_channels == 1:
            def wrapped(images: np.ndarray) -> np.ndarray:
                gray = images.mean(axis=-1, keepdims=True).astype(np.float32)
                return predict_fn(gray)
            return wrapped
        return predict_fn

    def _explain_image_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        from lime.lime_image import LimeImageExplainer as _LimeImage

        from .image_utils import lime_panel

        n_samples = min(self.config.global_samples, len(dataset))
        lime_img = _LimeImage()
        panels = []

        if model.task == "classification":
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            n_classes = len(classes)
            predict_fn = model.predict_proba

            for i in range(n_samples):
                image = dataset.instance(i)  # (H, W, C)
                image_rgb = self._to_rgb(image)
                wrapped_fn = self._wrap_predict_rgb(predict_fn, image.shape[-1])

                proba = predict_fn(image.reshape(1, *image.shape))[0]
                predicted_idx = int(np.argmax(proba))
                predicted_label = classes[predicted_idx]
                confidence = float(proba[predicted_idx])

                exp = lime_img.explain_instance(
                    image_rgb.astype(np.double),
                    wrapped_fn,
                    num_features=self.config.num_features,
                    num_samples=self.config.num_samples,
                    labels=list(range(n_classes)),
                )
                panels.append(lime_panel(
                    image_rgb, exp, predicted_idx, self.config.num_features,
                    title=f"Sample {i} — pred: {predicted_label}",
                    confidence=confidence,
                ))

            return ExplanationResult(
                mode="global", method="lime", task=model.task,
                target_name=dataset.target_name, contributions=[],
                base_values=None, global_samples=n_samples,
                image_panels=panels,
            )
        else:
            def predict_fn_reg(images: np.ndarray) -> np.ndarray:
                return model.predict(images).reshape(-1, 1)

            bg_sample = dataset.X[np.random.choice(len(dataset), size=min(100, len(dataset)), replace=False)]
            reg_base: dict[ty.Any, float] = {None: float(model.predict(bg_sample).mean())}  # type: ignore[dict-item]

            for i in range(n_samples):
                image = dataset.instance(i)  # (H, W, C)
                image_rgb = self._to_rgb(image)
                wrapped_reg = self._wrap_predict_rgb(predict_fn_reg, image.shape[-1])

                predicted_value = float(model.predict(image.reshape(1, *image.shape))[0])
                exp = lime_img.explain_instance(
                    image_rgb.astype(np.double),
                    wrapped_reg,
                    num_features=self.config.num_features,
                    num_samples=self.config.num_samples,
                    labels=[0],
                )
                panels.append(lime_panel(
                    image_rgb, exp, 0, self.config.num_features,
                    title=f"Sample {i} — pred: {predicted_value:.4f}",
                ))

            return ExplanationResult(
                mode="global", method="lime", task=model.task,
                target_name=dataset.target_name, contributions=[],
                base_values=reg_base, global_samples=n_samples,
                image_panels=panels,
            )

    def _explain_image_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        from lime.lime_image import LimeImageExplainer as _LimeImage

        from .image_utils import lime_panel

        lime_img = _LimeImage()
        image = instance.instance(0)  # (H, W, C)

        if model.task == "classification":
            predict_fn = model.predict_proba
            proba = predict_fn(image.reshape(1, *image.shape))[0]
            predicted_idx = int(np.argmax(proba))
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            predicted_label = classes[predicted_idx]
            confidence = float(proba[predicted_idx])
            n_classes = len(classes)

            image_rgb = self._to_rgb(image)
            wrapped_fn = self._wrap_predict_rgb(predict_fn, image.shape[-1])
            exp = lime_img.explain_instance(
                image_rgb.astype(np.double),
                wrapped_fn,
                num_features=self.config.num_features,
                num_samples=self.config.num_samples,
                labels=list(range(n_classes)),
            )
            contributions = [
                FeatureContribution(feature=f"region_{seg_id}", weight=weight, label=predicted_label)
                for seg_id, weight in exp.local_exp.get(predicted_idx, [])
            ]
            base_values: dict[ty.Any, float] = {predicted_label: confidence}
            panel = lime_panel(
                image_rgb, exp, predicted_idx, self.config.num_features,
                title=f"pred: {predicted_label} ({confidence:.2%})",
                confidence=confidence,
            )
        else:
            def predict_fn_reg(images: np.ndarray) -> np.ndarray:
                return model.predict(images).reshape(-1, 1)

            predicted_value = float(model.predict(image.reshape(1, *image.shape))[0])
            image_rgb = self._to_rgb(image)
            wrapped_reg = self._wrap_predict_rgb(predict_fn_reg, image.shape[-1])
            exp = lime_img.explain_instance(
                image_rgb.astype(np.double),
                wrapped_reg,
                num_features=self.config.num_features,
                num_samples=self.config.num_samples,
                labels=[0],
            )
            contributions = [
                FeatureContribution(feature=f"region_{seg_id}", weight=weight, label=None)
                for seg_id, weight in exp.local_exp.get(0, [])
            ]
            base_values = {None: predicted_value}
            panel = lime_panel(
                image_rgb, exp, 0, self.config.num_features,
                title=f"pred: {predicted_value:.4f}",
            )

        return ExplanationResult(
            mode="local", method="lime", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values,
            image_panels=[panel],
        )

    # ------------------------------------------------------------------
    # Internal helpers — text
    # ------------------------------------------------------------------

    def _explain_text_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        """LIME global explanation for text classification.

        Args:
            model: HfAdapter (or any adapter with predict_proba accepting list[str]).
            dataset: TextDataset with texts and class labels.

        Returns:
            ExplanationResult with image_panels and word-level contributions.
        """
        from lime.lime_text import LimeTextExplainer

        from pulsetrace.data.text_dataset import TextDataset

        from .text_utils import text_panel

        assert isinstance(dataset, TextDataset)

        predict_fn = model.predict_proba  # type: ignore[arg-type]
        n_samples = min(self.config.global_samples, len(dataset))

        bg_texts = dataset.texts[: min(20, len(dataset))]
        bg_proba = predict_fn(bg_texts)
        n_classes = bg_proba.shape[1]

        ds_classes = dataset.classes if dataset.classes is not None else np.array([])
        if len(ds_classes) == n_classes:
            class_names = [str(c) for c in ds_classes]
        else:
            class_names = [str(i) for i in range(n_classes)]

        base_values: dict[str | int | None, float] = {
            class_names[ci]: float(np.mean(bg_proba[:, ci])) for ci in range(n_classes)
        }

        lime_exp = LimeTextExplainer(class_names=class_names)
        totals: dict[ty.Any, dict[str, float]] = {cn: {} for cn in class_names}
        counts: dict[ty.Any, dict[str, int]] = {cn: {} for cn in class_names}
        panels = []

        for i in range(n_samples):
            text = dataset.texts[i]
            exp = lime_exp.explain_instance(
                text,
                predict_fn,
                num_features=self.config.num_features,
                labels=list(range(n_classes)),
            )
            proba = predict_fn([text])[0]
            predicted_idx = int(np.argmax(proba))
            predicted_label = class_names[predicted_idx]
            confidence = float(proba[predicted_idx])

            for idx, cn in enumerate(class_names):
                for feat, weight in exp.as_list(label=idx):
                    totals[cn][feat] = totals[cn].get(feat, 0.0) + weight
                    counts[cn][feat] = counts[cn].get(feat, 0) + 1

            word_weights = exp.as_list(label=predicted_idx)
            panels.append(text_panel(
                text, word_weights,
                title=f"Sample {i} — pred: {predicted_label}",
                confidence=confidence,
            ))

        contributions = [
            FeatureContribution(feature=feat, weight=totals[cn][feat] / counts[cn][feat], label=cn)
            for cn in class_names
            for feat in totals[cn]
        ]

        return ExplanationResult(
            mode="global", method="lime", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values, global_samples=n_samples,
            image_panels=panels,
        )

    def _explain_text_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        """LIME local explanation for text classification.

        Args:
            model: HfAdapter (or any adapter with predict_proba accepting list[str]).
            instance: TextDataset with a single text to explain.
            dataset: TextDataset used for background base values.

        Returns:
            ExplanationResult with one image_panel and word-level contributions.
        """
        from lime.lime_text import LimeTextExplainer

        from pulsetrace.data.text_dataset import TextDataset

        from .text_utils import text_panel

        assert isinstance(instance, TextDataset)
        assert isinstance(dataset, TextDataset)

        predict_fn = model.predict_proba  # type: ignore[arg-type]
        text = instance.texts[0]

        proba = predict_fn([text])[0]
        n_classes = len(proba)
        predicted_idx = int(np.argmax(proba))
        confidence = float(proba[predicted_idx])

        ds_classes = dataset.classes if dataset.classes is not None else np.array([])
        if len(ds_classes) == n_classes:
            class_names = [str(c) for c in ds_classes]
        else:
            class_names = [str(i) for i in range(n_classes)]
        predicted_label = class_names[predicted_idx]

        lime_exp = LimeTextExplainer(class_names=class_names)
        exp = lime_exp.explain_instance(
            text,
            predict_fn,
            num_features=self.config.num_features,
            labels=list(range(n_classes)),
        )

        word_weights = exp.as_list(label=predicted_idx)
        contributions = [
            FeatureContribution(feature=feat, weight=weight, label=predicted_label)
            for feat, weight in word_weights
        ]

        bg_texts = dataset.texts[: min(20, len(dataset))]
        bg_proba = predict_fn(bg_texts)
        base_values: dict[str | int | None, float] = {
            class_names[ci]: float(np.mean(bg_proba[:, ci])) for ci in range(n_classes)
        }

        panel = text_panel(
            text, word_weights,
            title=f"pred: {predicted_label}",
            confidence=confidence,
        )

        return ExplanationResult(
            mode="local", method="lime", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values,
            image_panels=[panel],
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        from pulsetrace.data.image_dataset import ImageDataset
        from pulsetrace.data.text_dataset import TextDataset
        from pulsetrace.data.timeseries_dataset import TimeSeriesDataset
        if isinstance(dataset, ImageDataset):
            return self._explain_image_global(model, dataset)
        if isinstance(dataset, TimeSeriesDataset):
            return self._explain_ts_global(model, dataset)
        if isinstance(dataset, TextDataset):
            return self._explain_text_global(model, dataset)
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
        from pulsetrace.data.text_dataset import TextDataset
        from pulsetrace.data.timeseries_dataset import TimeSeriesDataset
        if isinstance(dataset, ImageDataset):
            return self._explain_image_local(model, instance, dataset)
        if isinstance(dataset, TimeSeriesDataset):
            return self._explain_ts_local(model, instance, dataset)
        if isinstance(dataset, TextDataset):
            return self._explain_text_local(model, instance, dataset)
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
