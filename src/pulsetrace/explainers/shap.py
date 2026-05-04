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
        from .ts_utils import ts_panel

        background = dataset.X[:100] if len(dataset) > 100 else dataset.X
        explainer = self._make_kernel_explainer(model, background, dataset.feature_names)

        sample = dataset.X[:10] if len(dataset) > 10 else dataset.X
        shap_values = explainer.shap_values(sample, silent=True)

        contributions: list[FeatureContribution] = []
        base_values: dict[str | int | None, float] = {}
        panels = []

        if model.task == "classification":
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            sv_arr = np.array(shap_values)
            ev = explainer.expected_value
            if hasattr(ev, "__len__"):
                base_values = {int(cls): float(ev[i]) for i, cls in enumerate(classes[:len(ev)])}
            else:
                base_values = {None: float(ev)}

            if isinstance(shap_values, list):
                # Legacy list format: (n_classes, n_samples, n_features)
                for class_idx, cls in enumerate(classes):
                    avg_abs = np.mean(np.abs(sv_arr[class_idx]), axis=0)
                    contributions.extend(
                        self._top_contributions(avg_abs, dataset.feature_names, label=cls)
                    )
                for i in range(len(sample)):
                    proba = model.predict_proba(sample[i : i + 1])[0]
                    predicted_idx = int(np.argmax(proba))
                    predicted_label = classes[predicted_idx]
                    confidence = float(proba[predicted_idx])
                    panels.append(ts_panel(
                        sample[i], sv_arr[predicted_idx, i],
                        title=f"Sample {i} — pred: {predicted_label}",
                        confidence=confidence,
                    ))
            elif sv_arr.ndim == 3:
                # New ndarray format: (n_samples, n_features, n_classes)
                n_cls = sv_arr.shape[2]
                for class_idx, cls in enumerate(classes[:n_cls]):
                    avg_abs = np.mean(np.abs(sv_arr[:, :, class_idx]), axis=0)
                    contributions.extend(
                        self._top_contributions(avg_abs, dataset.feature_names, label=cls)
                    )
                for i in range(len(sample)):
                    proba = model.predict_proba(sample[i : i + 1])[0]
                    predicted_idx = int(np.argmax(proba))
                    predicted_label = classes[predicted_idx]
                    confidence = float(proba[predicted_idx])
                    panels.append(ts_panel(
                        sample[i], sv_arr[i, :, predicted_idx],
                        title=f"Sample {i} — pred: {predicted_label}",
                        confidence=confidence,
                    ))
            else:
                avg_abs = np.mean(np.abs(sv_arr), axis=0)
                contributions.extend(
                    self._top_contributions(avg_abs, dataset.feature_names, label=None)
                )
        else:
            sv_arr = np.array(shap_values)
            avg_abs = np.mean(np.abs(sv_arr), axis=0)
            contributions.extend(
                self._top_contributions(avg_abs, dataset.feature_names, label=None)
            )
            ev = explainer.expected_value
            base_val_reg = float(ev) if np.ndim(ev) == 0 else float(ev[0])
            base_values[None] = base_val_reg
            for i in range(len(sample)):
                predicted_value = float(model.predict(sample[i : i + 1])[0])
                weights_i = sv_arr[i] if sv_arr.ndim >= 2 else sv_arr
                panels.append(ts_panel(
                    sample[i], weights_i,
                    title=f"Sample {i} — pred: {predicted_value:.4f}",
                ))

        return ExplanationResult(
            mode="global", method="shap", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values, global_samples=len(sample),
            image_panels=panels if panels else None,
        )

    def _explain_ts_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        from .ts_utils import ts_panel

        background = dataset.X[:100] if len(dataset) > 100 else dataset.X
        explainer = self._make_kernel_explainer(model, background, dataset.feature_names)

        x = instance.instance(0).reshape(1, -1)
        shap_values = explainer.shap_values(x, silent=True)

        contributions: list[FeatureContribution] = []
        base_values: dict[str | int | None, float] = {}

        if model.task == "classification":
            proba = model.predict_proba(x)[0]
            predicted_idx = int(np.argmax(proba))
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            predicted_label = classes[predicted_idx]
            confidence = float(proba[predicted_idx])
            sv_arr = np.array(shap_values)
            if isinstance(shap_values, list):
                vals = sv_arr[predicted_idx][0]
            elif sv_arr.ndim == 3:
                vals = sv_arr[0, :, predicted_idx]
            else:
                vals = sv_arr[0]
            contributions.extend(
                self._top_contributions(vals, dataset.feature_names, label=predicted_label)
            )
            ev = explainer.expected_value
            if hasattr(ev, "__len__"):
                base_values = {int(cls): float(ev[i]) for i, cls in enumerate(classes[:len(ev)])}
            else:
                base_values = {None: float(ev)}
            panel = ts_panel(
                x[0], vals,
                title=f"pred: {predicted_label}",
                confidence=confidence,
            )
        else:
            sv_arr = np.array(shap_values)
            vals = sv_arr[0] if sv_arr.ndim >= 2 else sv_arr
            contributions.extend(
                self._top_contributions(vals, dataset.feature_names, label=None)
            )
            ev = explainer.expected_value
            base_val = float(ev) if np.ndim(ev) == 0 else float(ev[0])
            base_values[None] = base_val
            predicted_value = float(model.predict(x)[0])
            panel = ts_panel(
                x[0], vals,
                title=f"pred: {predicted_value:.4f}",
            )

        return ExplanationResult(
            mode="local", method="shap", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values,
            image_panels=[panel],
        )

    # ------------------------------------------------------------------
    # Internal helpers — image
    # ------------------------------------------------------------------

    def _explain_image_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        from .image_utils import shap_panel

        sample = dataset.X[:min(10, len(dataset))]   # (n, H, W, C)
        bg_mean = sample.mean(axis=0)                 # (H, W, C) — fill value for masker

        masker = shap.maskers.Image(bg_mean, bg_mean.shape)
        predict_fn = model.predict_proba if model.task == "classification" else model.predict
        explainer = shap.Explainer(predict_fn, masker)

        shap_values = explainer(sample)
        vals = shap_values.values
        # classification (predict_proba → (n, k)): vals.shape == (n, H, W, C, k)
        # regression     (predict → (n,)):         vals.shape == (n, H, W, C)

        panels = []
        base_values: dict[str | int | None, float] = {}

        if model.task == "classification":
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            for i in range(len(sample)):
                proba = predict_fn(sample[i : i + 1])[0]
                predicted_idx = int(np.argmax(proba))
                predicted_label = classes[predicted_idx]
                confidence = float(proba[predicted_idx])
                class_vals = vals[i, ..., predicted_idx]  # (H, W, C)
                panels.append(shap_panel(
                    sample[i], class_vals,
                    title=f"Sample {i} — pred: {predicted_label}",
                    confidence=confidence,
                ))
            ev = shap_values.base_values
            base_values[None] = float(np.mean(ev)) if ev is not None else 0.0
        else:
            for i in range(len(sample)):
                image_vals = vals[i]   # (H, W, C)
                predicted_value = float(predict_fn(sample[i : i + 1])[0])
                panels.append(shap_panel(
                    sample[i], image_vals,
                    title=f"Sample {i} — pred: {predicted_value:.4f}",
                ))
            ev = shap_values.base_values
            base_values[None] = float(np.mean(ev)) if ev is not None else 0.0

        return ExplanationResult(
            mode="global", method="shap", task=model.task,
            target_name=dataset.target_name, contributions=[],
            base_values=base_values, global_samples=len(sample),
            image_panels=panels,
        )

    def _explain_image_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        import skimage.segmentation

        from .image_utils import shap_panel

        image = instance.instance(0)                                      # (H, W, C)
        bg_mean = dataset.X[:min(10, len(dataset))].mean(axis=0)          # (H, W, C)

        masker = shap.maskers.Image(bg_mean, bg_mean.shape)
        predict_fn = model.predict_proba if model.task == "classification" else model.predict
        explainer = shap.Explainer(predict_fn, masker)

        shap_values = explainer(image.reshape(1, *image.shape))
        vals = shap_values.values   # (1, H, W, C) regression  OR  (1, H, W, C, n_classes)

        n_ch = image.shape[-1] if image.ndim == 3 else 1
        segments = skimage.segmentation.quickshift(
            image, kernel_size=4, max_dist=200, ratio=0.2,
            convert2lab=(n_ch == 3),
        )

        contributions: list[FeatureContribution] = []
        base_values: dict[str | int | None, float] = {}

        if model.task == "classification":
            proba = predict_fn(image.reshape(1, *image.shape))[0]
            predicted_idx = int(np.argmax(proba))
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            predicted_label = classes[predicted_idx]
            confidence = float(proba[predicted_idx])

            class_vals = vals[0, ..., predicted_idx]       # (H, W, C)
            avg_abs_hw = np.abs(class_vals).mean(axis=-1)  # (H, W)
            region_weights: dict[int, float] = {
                int(sid): float(avg_abs_hw[segments == sid].mean())
                for sid in np.unique(segments)
            }
            top = sorted(region_weights.items(), key=lambda x: x[1], reverse=True)[: self.config.num_features]
            for seg_id, w in top:
                contributions.append(FeatureContribution(feature=f"region_{seg_id}", weight=w, label=predicted_label))
            base_values[predicted_label] = confidence
            panel = shap_panel(
                image, class_vals,
                title=f"pred: {predicted_label} ({confidence:.2%})",
                confidence=confidence,
            )
        else:
            image_vals = vals[0]                               # (H, W, C)
            predicted_value = float(predict_fn(image.reshape(1, *image.shape))[0])
            avg_abs_hw = np.abs(image_vals).mean(axis=-1)      # (H, W)
            region_weights_r: dict[int, float] = {
                int(sid): float(avg_abs_hw[segments == sid].mean())
                for sid in np.unique(segments)
            }
            top_r = sorted(region_weights_r.items(), key=lambda x: x[1], reverse=True)[: self.config.num_features]
            for seg_id, w in top_r:
                contributions.append(FeatureContribution(feature=f"region_{seg_id}", weight=w, label=None))
            ev = shap_values.base_values
            base_values[None] = float(np.mean(ev)) if ev is not None else 0.0
            panel = shap_panel(image, image_vals, title=f"pred: {predicted_value:.4f}")

        return ExplanationResult(
            mode="local", method="shap", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values,
            image_panels=[panel],
        )

    # ------------------------------------------------------------------
    # Internal helpers — text
    # ------------------------------------------------------------------

    def _explain_text_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        """SHAP global explanation for text classification.

        Args:
            model: HfAdapter with predict_proba(list[str]) and tokenizer attribute.
            dataset: TextDataset with texts and class labels.

        Returns:
            ExplanationResult with image_panels and token-level contributions.
        """
        from pulsetrace.data.text_dataset import TextDataset

        from .text_utils import text_panel

        assert isinstance(dataset, TextDataset)

        n_samples = min(self.config.global_samples, len(dataset))
        texts = dataset.texts[:n_samples]

        masker = shap.maskers.Text(tokenizer=model.tokenizer)  # type: ignore[attr-defined]
        explainer = shap.Explainer(model.predict_proba, masker)  # type: ignore[arg-type]
        shap_values = explainer(texts, silent=True)

        sv: np.ndarray = shap_values.values  # (n, n_tokens, n_classes) or (n, n_tokens)
        n_classes = sv.shape[2] if sv.ndim == 3 else 1

        ds_classes = dataset.classes if dataset.classes is not None else np.array([])
        if len(ds_classes) == n_classes:
            class_names = [str(c) for c in ds_classes]
        else:
            class_names = [str(i) for i in range(n_classes)]

        ev = explainer.expected_value
        if hasattr(ev, "__len__"):
            base_values: dict[str | int | None, float] = {
                class_names[ci]: float(ev[ci]) for ci in range(min(len(ev), n_classes))
            }
        else:
            base_values = {class_names[0]: float(ev)}

        panels = []
        totals: dict[str, dict[str, float]] = {cn: {} for cn in class_names}
        counts: dict[str, dict[str, int]] = {cn: {} for cn in class_names}

        for i in range(n_samples):
            tokens = shap_values.data[i]
            proba = model.predict_proba([texts[i]])[0]  # type: ignore[arg-type]
            predicted_idx = int(np.argmax(proba[:n_classes]))
            predicted_label = class_names[predicted_idx]
            confidence = float(proba[predicted_idx])

            weights_for_pred = sv[i, :, predicted_idx] if sv.ndim == 3 else sv[i, :]
            word_weights = [(str(tok), float(w)) for tok, w in zip(tokens, weights_for_pred)]

            panels.append(text_panel(
                texts[i], word_weights,
                title=f"Sample {i} — pred: {predicted_label}",
                confidence=confidence,
            ))

            for tok, w in zip(tokens, weights_for_pred):
                tok_str = str(tok)
                totals[predicted_label][tok_str] = (
                    totals[predicted_label].get(tok_str, 0.0) + float(abs(w))
                )
                counts[predicted_label][tok_str] = counts[predicted_label].get(tok_str, 0) + 1

        all_contribs = [
            FeatureContribution(
                feature=feat,
                weight=totals[cn][feat] / counts[cn][feat],
                label=cn,
            )
            for cn in class_names
            for feat in totals[cn]
        ]
        contributions = sorted(all_contribs, key=lambda fc: abs(fc.weight), reverse=True)[
            : self.config.num_features * max(len(class_names), 1)
        ]

        return ExplanationResult(
            mode="global", method="shap", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values, global_samples=n_samples,
            image_panels=panels,
        )

    def _explain_text_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        """SHAP local explanation for text classification.

        Args:
            model: HfAdapter with predict_proba(list[str]) and tokenizer attribute.
            instance: TextDataset with a single text to explain.
            dataset: TextDataset used to derive class names.

        Returns:
            ExplanationResult with one image_panel and token-level contributions.
        """
        from pulsetrace.data.text_dataset import TextDataset

        from .text_utils import text_panel

        assert isinstance(instance, TextDataset)
        assert isinstance(dataset, TextDataset)

        text = instance.texts[0]

        masker = shap.maskers.Text(tokenizer=model.tokenizer)  # type: ignore[attr-defined]
        explainer = shap.Explainer(model.predict_proba, masker)  # type: ignore[arg-type]
        shap_values = explainer([text], silent=True)

        sv: np.ndarray = shap_values.values  # (1, n_tokens, n_classes) or (1, n_tokens)
        n_classes = sv.shape[2] if sv.ndim == 3 else 1
        tokens = shap_values.data[0]

        ds_classes = dataset.classes if dataset.classes is not None else np.array([])
        if len(ds_classes) == n_classes:
            class_names = [str(c) for c in ds_classes]
        else:
            class_names = [str(i) for i in range(n_classes)]

        proba = model.predict_proba([text])[0]  # type: ignore[arg-type]
        predicted_idx = int(np.argmax(proba[:n_classes]))
        predicted_label = class_names[predicted_idx]
        confidence = float(proba[predicted_idx])

        weights = sv[0, :, predicted_idx] if sv.ndim == 3 else sv[0, :]
        word_weights = [(str(tok), float(w)) for tok, w in zip(tokens, weights)]

        top_pairs = sorted(zip(tokens, weights), key=lambda tw: abs(tw[1]), reverse=True)[
            : self.config.num_features
        ]
        contributions = [
            FeatureContribution(feature=str(tok), weight=float(w), label=predicted_label)
            for tok, w in top_pairs
        ]

        ev = explainer.expected_value
        if hasattr(ev, "__len__"):
            base_values: dict[str | int | None, float] = {
                class_names[ci]: float(ev[ci]) for ci in range(min(len(ev), n_classes))
            }
        else:
            base_values = {class_names[0]: float(ev)}

        panel = text_panel(
            text, word_weights,
            title=f"pred: {predicted_label}",
            confidence=confidence,
        )

        return ExplanationResult(
            mode="local", method="shap", task=model.task,
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
