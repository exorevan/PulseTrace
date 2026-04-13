from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from .base import ModelAdapter

if TYPE_CHECKING:
    import torch.nn as nn


class PyTorchAdapter(ModelAdapter):
    """Wraps a PyTorch nn.Module.

    Task must be provided explicitly — there is no reliable way to
    auto-detect classification vs regression from a PyTorch model graph.
    """

    def __init__(
        self,
        model: "nn.Module",
        task: Literal["classification", "regression"],
    ) -> None:
        self._model = model
        self.task = task

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        import torch

        self._model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X)
            out = self._model(t).numpy()
        if self.task == "regression":
            return out.flatten()
        return np.argmax(out, axis=1)

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.task == "regression":
            raise NotImplementedError("predict_proba is not available for regression models.")
        import torch
        import torch.nn.functional as F

        self._model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X)
            out = self._model(t)
            return F.softmax(out, dim=1).numpy()
