from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from .base import ModelAdapter


class SklearnAdapter(ModelAdapter):
    """Wraps a scikit-learn estimator."""

    def __init__(self, model: object) -> None:
        self._model = model
        self.task: Literal["classification", "regression"] = (
            "classification" if hasattr(model, "predict_proba") else "regression"
        )
        self._feature_names: list[str] | None = (
            list(model.feature_names_in_)  # type: ignore[union-attr]
            if hasattr(model, "feature_names_in_")
            else None
        )

    def _as_frame(self, X: npt.NDArray[np.float64]) -> pd.DataFrame | npt.NDArray[np.float64]:
        if self._feature_names is not None and X.ndim == 2:
            return pd.DataFrame(X, columns=self._feature_names)
        return X

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self._model.predict(self._as_frame(X))  # type: ignore[union-attr]

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.task == "regression":
            raise NotImplementedError("predict_proba is not available for regression models.")
        return self._model.predict_proba(self._as_frame(X))  # type: ignore[union-attr]
