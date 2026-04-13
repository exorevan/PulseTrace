from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from .base import ModelAdapter

if TYPE_CHECKING:
    import keras


class KerasAdapter(ModelAdapter):
    """Wraps a Keras model.

    Task is inferred from the number of output units:
      1 output unit  → regression
      >1 output units → classification
    """

    def __init__(self, model: "keras.Model") -> None:
        self._model = model
        output_units: int = model.output_shape[-1]
        self.task: Literal["classification", "regression"] = (
            "regression" if output_units == 1 else "classification"
        )
        self._n_classes = output_units if self.task == "classification" else 0

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raw: npt.NDArray[np.float64] = self._model.predict(X, verbose=0)
        if self.task == "regression":
            return raw.flatten()
        return np.argmax(raw, axis=1)

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.task == "regression":
            raise NotImplementedError("predict_proba is not available for regression models.")
        return self._model.predict(X, verbose=0)
