"""Integrated Gradients explainer — PyTorch and HuggingFace backends only."""
from __future__ import annotations

import typing as ty

import numpy as np

from pulsetrace.adapters.base import ModelAdapter
from pulsetrace.config.schema import ExplainerConfig
from pulsetrace.data.dataset import Dataset

from .base import BaseExplainer
from .result import ExplanationResult, FeatureContribution, ImagePanel


class IgExplainer(BaseExplainer):
    """Integrated Gradients explanations for PyTorch and HuggingFace models."""

    def __init__(self, config: ExplainerConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Guards and shared helpers
    # ------------------------------------------------------------------

    def _check_backend(self, model: ModelAdapter) -> None:
        from pulsetrace.adapters.huggingface import HfAdapter
        from pulsetrace.adapters.pytorch import PyTorchAdapter

        if isinstance(model, PyTorchAdapter):
            return
        if isinstance(model, HfAdapter):
            return
        from pulsetrace.adapters.keras import KerasAdapter
        if isinstance(model, KerasAdapter):
            raise NotImplementedError(
                "Integrated Gradients requires a gradient-capable backend "
                "(pt or hf). Keras support is planned."
            )
        raise NotImplementedError(
            "Integrated Gradients requires a gradient-capable backend (pt or hf)."
        )

    def _top_contributions(
        self,
        weights: np.ndarray,
        feature_names: list[str],
        label: str | int | None,
    ) -> list[FeatureContribution]:
        weights = np.asarray(weights).ravel()
        top_indices = np.argsort(np.abs(weights))[::-1][: self.config.num_features]
        return [
            FeatureContribution(
                feature=feature_names[int(i)],
                weight=float(weights[int(i)]),
                label=label,
            )
            for i in top_indices
        ]

    def _average_contributions(
        self,
        all_contribs: list[list[FeatureContribution]],
    ) -> list[FeatureContribution]:
        totals: dict[tuple[str, ty.Any], float] = {}
        counts: dict[tuple[str, ty.Any], int] = {}
        for contribs in all_contribs:
            for fc in contribs:
                key = (fc.feature, fc.label)
                totals[key] = totals.get(key, 0.0) + fc.weight
                counts[key] = counts.get(key, 0) + 1
        return [
            FeatureContribution(
                feature=feat,
                weight=totals[(feat, lbl)] / counts[(feat, lbl)],
                label=lbl,
            )
            for feat, lbl in totals
        ]

    def _single_instance_dataset(self, dataset: Dataset, i: int) -> Dataset:
        y_slice = dataset.y[i : i + 1] if len(dataset.y) > 0 else np.array([])
        return Dataset(
            X=dataset.X[i : i + 1],
            y=y_slice,
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            classes=dataset.classes,
        )

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    def explain_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        self._check_backend(model)
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

    def explain_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        self._check_backend(model)
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

    # ------------------------------------------------------------------
    # Stubs — implemented in subsequent tasks
    # ------------------------------------------------------------------

    def _explain_tabular_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        import torch
        from captum.attr import IntegratedGradients

        nn_model = model._model  # type: ignore[attr-defined]
        task = model.task

        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            out = nn_model(x)
            if task == "classification":
                return torch.softmax(out, dim=-1)
            return out.squeeze(-1)

        x_np = instance.instance(0).astype(np.float32)
        x_tensor = torch.FloatTensor(x_np).unsqueeze(0)  # (1, n_features)

        if self.config.ig_baseline == "mean":
            baseline = torch.FloatTensor(
                dataset.X.mean(axis=0).astype(np.float32)
            ).unsqueeze(0)
        else:
            baseline = torch.zeros_like(x_tensor)

        ig = IntegratedGradients(forward_fn)
        contributions: list[FeatureContribution] = []
        base_values: dict[str | int | None, float] = {}

        if task == "classification":
            with torch.no_grad():
                proba = torch.softmax(nn_model(x_tensor), dim=-1)[0].numpy()
            predicted_idx = int(np.argmax(proba))
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            predicted_label = classes[predicted_idx]

            attrs = ig.attribute(
                x_tensor, baseline, target=predicted_idx, n_steps=self.config.ig_steps
            )
            weights = attrs[0].detach().numpy()  # (n_features,)
            contributions = self._top_contributions(
                weights, dataset.feature_names, predicted_label
            )
            base_values[predicted_label] = float(proba[predicted_idx])
        else:
            with torch.no_grad():
                predicted_value = float(nn_model(x_tensor).squeeze().numpy())

            attrs = ig.attribute(x_tensor, baseline, n_steps=self.config.ig_steps)
            weights = attrs[0].detach().numpy()
            contributions = self._top_contributions(weights, dataset.feature_names, None)
            base_values[None] = predicted_value

        return ExplanationResult(
            mode="local", method="ig", task=task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values,
        )

    def _explain_tabular_global(
        self, model: ModelAdapter, dataset: Dataset
    ) -> ExplanationResult:
        n_samples = min(self.config.global_samples, len(dataset))
        all_contribs: list[list[FeatureContribution]] = []
        base_values: dict[str | int | None, float] = {}

        for i in range(n_samples):
            single = self._single_instance_dataset(dataset, i)
            local = self._explain_tabular_local(model, single, dataset)
            all_contribs.append(local.contributions)
            if not base_values and local.base_values:
                base_values = local.base_values

        return ExplanationResult(
            mode="global", method="ig", task=model.task,
            target_name=dataset.target_name,
            contributions=self._average_contributions(all_contribs),
            base_values=base_values or None,
            global_samples=n_samples,
        )

    def _explain_image_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        import torch
        from captum.attr import IntegratedGradients

        from .image_utils import shap_panel

        nn_model = model._model  # type: ignore[attr-defined]
        task = model.task
        image = instance.instance(0)  # (H, W, C)

        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            out = nn_model(x)
            if task == "classification":
                return torch.softmax(out, dim=-1)
            return out.squeeze(-1)

        x_tensor = torch.FloatTensor(image).unsqueeze(0)  # (1, H, W, C)

        if self.config.ig_baseline == "mean":
            bg = dataset.X[: min(100, len(dataset))].mean(axis=0).astype(np.float32)
            baseline = torch.FloatTensor(bg).unsqueeze(0)
        else:
            baseline = torch.zeros_like(x_tensor)

        ig = IntegratedGradients(forward_fn)
        contributions: list[FeatureContribution] = []
        base_values: dict[str | int | None, float] = {}

        if task == "classification":
            with torch.no_grad():
                proba = torch.softmax(nn_model(x_tensor), dim=-1)[0].numpy()
            predicted_idx = int(np.argmax(proba))
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            predicted_label = classes[predicted_idx]
            confidence = float(proba[predicted_idx])

            attrs = ig.attribute(
                x_tensor, baseline, target=predicted_idx, n_steps=self.config.ig_steps
            )
            attr_np = attrs[0].detach().numpy()  # (H, W, C)
            base_values[predicted_label] = confidence
            panel = shap_panel(
                image, attr_np,
                title=f"pred: {predicted_label} ({confidence:.2%})",
                confidence=confidence,
            )
        else:
            with torch.no_grad():
                predicted_value = float(nn_model(x_tensor).squeeze().numpy())

            attrs = ig.attribute(x_tensor, baseline, n_steps=self.config.ig_steps)
            attr_np = attrs[0].detach().numpy()  # (H, W, C)
            base_values[None] = predicted_value
            panel = shap_panel(image, attr_np, title=f"pred: {predicted_value:.4f}")

        return ExplanationResult(
            mode="local", method="ig", task=task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values, image_panels=[panel],
        )

    def _explain_image_global(
        self, model: ModelAdapter, dataset: Dataset
    ) -> ExplanationResult:
        n_samples = min(self.config.global_samples, len(dataset))
        panels = []
        base_values: dict[str | int | None, float] = {}

        for i in range(n_samples):
            single = self._single_instance_dataset(dataset, i)
            local = self._explain_image_local(model, single, dataset)
            if local.image_panels:
                p = local.image_panels[0]
                panels.append(ImagePanel(
                    title=f"Sample {i} — {p.title}",
                    original_b64=p.original_b64,
                    explanation_b64=p.explanation_b64,
                    confidence=p.confidence,
                ))
            if not base_values and local.base_values:
                base_values = local.base_values

        return ExplanationResult(
            mode="global", method="ig", task=model.task,
            target_name=dataset.target_name, contributions=[],
            base_values=base_values or None, global_samples=n_samples,
            image_panels=panels if panels else None,
        )

    def _explain_ts_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        import torch
        from captum.attr import IntegratedGradients

        from .ts_utils import ts_panel

        nn_model = model._model  # type: ignore[attr-defined]
        task = model.task
        x_1d = instance.instance(0).astype(np.float32)  # (T,)

        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            out = nn_model(x)
            if task == "classification":
                return torch.softmax(out, dim=-1)
            return out.squeeze(-1)

        x_tensor = torch.FloatTensor(x_1d).unsqueeze(0)  # (1, T)

        if self.config.ig_baseline == "mean":
            baseline = torch.FloatTensor(
                dataset.X.mean(axis=0).astype(np.float32)
            ).unsqueeze(0)
        else:
            baseline = torch.zeros_like(x_tensor)

        ig = IntegratedGradients(forward_fn)
        contributions: list[FeatureContribution] = []
        base_values: dict[str | int | None, float] = {}

        if task == "classification":
            with torch.no_grad():
                proba = torch.softmax(nn_model(x_tensor), dim=-1)[0].numpy()
            predicted_idx = int(np.argmax(proba))
            classes = dataset.classes if dataset.classes is not None else np.array([0])
            predicted_label = classes[predicted_idx]
            confidence = float(proba[predicted_idx])

            attrs = ig.attribute(
                x_tensor, baseline, target=predicted_idx, n_steps=self.config.ig_steps
            )
            weights = attrs[0].detach().numpy()  # (T,)
            contributions = self._top_contributions(
                weights, dataset.feature_names, predicted_label
            )
            base_values[predicted_label] = confidence
            panel = ts_panel(
                x_1d, weights,
                title=f"pred: {predicted_label}",
                confidence=confidence,
            )
        else:
            with torch.no_grad():
                predicted_value = float(nn_model(x_tensor).squeeze().numpy())

            attrs = ig.attribute(x_tensor, baseline, n_steps=self.config.ig_steps)
            weights = attrs[0].detach().numpy()
            contributions = self._top_contributions(weights, dataset.feature_names, None)
            base_values[None] = predicted_value
            panel = ts_panel(x_1d, weights, title=f"pred: {predicted_value:.4f}")

        return ExplanationResult(
            mode="local", method="ig", task=task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values, image_panels=[panel],
        )

    def _explain_ts_global(
        self, model: ModelAdapter, dataset: Dataset
    ) -> ExplanationResult:
        n_samples = min(self.config.global_samples, len(dataset))
        all_contribs: list[list[FeatureContribution]] = []
        panels = []
        base_values: dict[str | int | None, float] = {}

        for i in range(n_samples):
            single = self._single_instance_dataset(dataset, i)
            local = self._explain_ts_local(model, single, dataset)
            all_contribs.append(local.contributions)
            if local.image_panels:
                p = local.image_panels[0]
                panels.append(ImagePanel(
                    title=f"Sample {i} — {p.title}",
                    original_b64=p.original_b64,
                    explanation_b64=p.explanation_b64,
                    confidence=p.confidence,
                ))
            if not base_values and local.base_values:
                base_values = local.base_values

        return ExplanationResult(
            mode="global", method="ig", task=model.task,
            target_name=dataset.target_name,
            contributions=self._average_contributions(all_contribs),
            base_values=base_values or None,
            global_samples=n_samples,
            image_panels=panels if panels else None,
        )

    def _explain_text_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        import torch
        from captum.attr import LayerIntegratedGradients

        from pulsetrace.data.text_dataset import TextDataset

        from .text_utils import text_panel

        assert isinstance(instance, TextDataset)
        assert isinstance(dataset, TextDataset)

        hf_model = model._model  # type: ignore[attr-defined]
        tokenizer = model.tokenizer  # type: ignore[attr-defined]
        max_length = model._max_length  # type: ignore[attr-defined]

        text = instance.texts[0]
        encoded = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids: torch.Tensor = encoded["input_ids"]
        attention_mask: torch.Tensor = encoded["attention_mask"]
        pad_id = tokenizer.pad_token_id or 0
        baseline_ids = torch.full_like(input_ids, pad_id)

        with torch.no_grad():
            logits = hf_model(**encoded).logits
        proba = torch.softmax(logits, dim=-1)[0].numpy()
        predicted_idx = int(np.argmax(proba))
        confidence = float(proba[predicted_idx])
        n_classes = len(proba)

        ds_classes = dataset.classes if dataset.classes is not None else np.array([])
        class_names = (
            [str(c) for c in ds_classes]
            if len(ds_classes) == n_classes
            else [str(i) for i in range(n_classes)]
        )
        predicted_label = class_names[predicted_idx]

        def forward_fn(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            out = hf_model(input_ids=ids, attention_mask=mask)
            return torch.softmax(out.logits, dim=-1)

        lig = LayerIntegratedGradients(forward_fn, hf_model.get_input_embeddings())
        attrs = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=predicted_idx,
            n_steps=self.config.ig_steps,
        )
        per_token = attrs[0].sum(dim=-1).detach().numpy()  # (seq_len,) — signed

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        word_weights = [(str(tok), float(w)) for tok, w in zip(tokens, per_token)]

        top_pairs = sorted(zip(tokens, per_token), key=lambda tw: abs(tw[1]), reverse=True)[
            : self.config.num_features
        ]
        contributions = [
            FeatureContribution(feature=str(tok), weight=float(w), label=predicted_label)
            for tok, w in top_pairs
        ]
        base_values: dict[str | int | None, float] = {
            class_names[ci]: float(proba[ci]) for ci in range(n_classes)
        }
        panel = text_panel(text, word_weights, title=f"pred: {predicted_label}", confidence=confidence)

        return ExplanationResult(
            mode="local", method="ig", task=model.task,
            target_name=dataset.target_name, contributions=contributions,
            base_values=base_values, image_panels=[panel],
        )

    def _explain_text_global(
        self, model: ModelAdapter, dataset: Dataset
    ) -> ExplanationResult:
        from pulsetrace.data.text_dataset import TextDataset

        assert isinstance(dataset, TextDataset)

        n_samples = min(self.config.global_samples, len(dataset))
        all_contribs: list[list[FeatureContribution]] = []
        panels = []
        base_values: dict[str | int | None, float] = {}

        for i in range(n_samples):
            single = TextDataset(
                X=np.zeros((1, 1), dtype=np.float32),
                y=dataset.y[i : i + 1],
                feature_names=dataset.feature_names,
                target_name=dataset.target_name,
                classes=dataset.classes,
                data_type="text",
                texts=[dataset.texts[i]],
            )
            local = self._explain_text_local(model, single, dataset)
            all_contribs.append(local.contributions)
            if local.image_panels:
                p = local.image_panels[0]
                panels.append(ImagePanel(
                    title=f"Sample {i} — {p.title}",
                    original_b64=p.original_b64,
                    explanation_b64=p.explanation_b64,
                    confidence=p.confidence,
                ))
            if not base_values and local.base_values:
                base_values = local.base_values

        return ExplanationResult(
            mode="global", method="ig", task=model.task,
            target_name=dataset.target_name,
            contributions=self._average_contributions(all_contribs),
            base_values=base_values or None,
            global_samples=n_samples,
            image_panels=panels if panels else None,
        )
