from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Dataset:
    """Immutable container for feature matrix, targets, and metadata."""

    X: npt.NDArray[np.float64]        # (n_samples, n_features)
    y: npt.NDArray[np.object_]        # (n_samples,) — empty array when only_x=True
    feature_names: list[str]
    target_name: str
    classes: npt.NDArray[np.object_] | None = None  # None when y is empty

    def __len__(self) -> int:
        return len(self.X)

    def instance(self, i: int) -> npt.NDArray[np.float64]:
        """Return the i-th row as a 1-D array."""
        return self.X[i]
