from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from .base import ModelAdapter


class SklearnAdapter(ModelAdapter):
    """Wraps a scikit-learn estimator."""

    def __init__(self, model: object) -> None:
        self._model = model
        self.task: Literal["classification", "regression"] = (
            "classification" if hasattr(model, "predict_proba") else "regression"
        )

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self._model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.task == "regression":
            raise NotImplementedError("predict_proba is not available for regression models.")
        return self._model.predict_proba(X)  # type: ignore[union-attr]
