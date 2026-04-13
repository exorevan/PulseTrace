from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import numpy.typing as npt


class ModelAdapter(ABC):
    """Unified interface for any ML model backend.

    Wraps sklearn, Keras, or PyTorch models so that explainers
    never need to import or inspect the underlying framework.
    """

    task: Literal["classification", "regression"]

    @abstractmethod
    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return predictions.

        Returns shape (n_samples,): class indices for classification,
        continuous values for regression.
        """

    @abstractmethod
    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return class probabilities of shape (n_samples, n_classes).

        Raises NotImplementedError for regression models.
        """
