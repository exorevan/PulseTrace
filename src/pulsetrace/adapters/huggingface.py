"""HuggingFace adapter — wraps AutoModelForSequenceClassification."""
from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from pulsetrace.config.schema import HfModelConfig

from .base import ModelAdapter


class HfAdapter(ModelAdapter):
    """Adapter for HuggingFace sequence classification models.

    Accepts a Hub model ID (e.g. "distilbert-base-uncased-finetuned-sst-2-english")
    or a local directory saved via model.save_pretrained().

    predict_proba and predict accept list[str] instead of numpy arrays.
    The text explainers call these methods directly; tabular explainers are never
    invoked on text datasets.
    """

    def __init__(self, config: HfModelConfig) -> None:
        """Load tokenizer and model from Hub or local path.

        Args:
            config: HfModelConfig specifying path_or_name, labels, and max_length.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.task: Literal["classification", "regression"] = "classification"
        self.labels: list[str] = config.labels
        self._max_length: int = config.max_length
        self.tokenizer = AutoTokenizer.from_pretrained(config.path_or_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(config.path_or_name)
        self._model.eval()

    def predict_proba(self, texts: list[str]) -> npt.NDArray[np.float64]:  # type: ignore[override]
        """Tokenize texts, run forward pass, return softmax probabilities (n, n_classes).

        Args:
            texts: Input strings to classify.

        Returns:
            Float64 array of shape (n, n_classes) summing to 1 per row.
        """
        import torch

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()
        return probs.astype(np.float64)

    def predict(self, texts: list[str]) -> npt.NDArray[np.float64]:  # type: ignore[override]
        """Return argmax class index per sample, shape (n,).

        Args:
            texts: Input strings to classify.

        Returns:
            Integer class indices as float64 array of shape (n,).
        """
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1).astype(np.float64)
