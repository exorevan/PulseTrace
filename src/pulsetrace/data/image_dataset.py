from __future__ import annotations

from dataclasses import dataclass

from .dataset import Dataset


@dataclass(frozen=True)
class ImageDataset(Dataset):
    """Dataset for images. X has shape (n_samples, H, W, C).

    Inherits all fields from Dataset. instance(i) returns X[i] which is (H, W, C).
    """

    @property
    def image_shape(self) -> tuple[int, ...]:
        """Returns (H, W, C) — the shape of a single image."""
        return tuple(self.X.shape[1:])
