from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from pulsetrace.config.schema import ImageDatasetConfig

from .image_dataset import ImageDataset

_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_image_dataset(config: ImageDatasetConfig) -> ImageDataset:
    """Load an image directory into an ImageDataset.

    Label convention: subdirectory name = class label (ImageNet-style).
    If only_x=True, all images in the directory are loaded with no labels.
    """
    root = Path(config.path)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected a directory: {root}")

    if config.only_x:
        images = _collect_flat(root, config)
        y: np.ndarray = np.array([], dtype=object)
        classes = None
        target_name = ""
    else:
        images, labels = _collect_labelled(root, config)
        y = np.array(labels, dtype=object)
        classes = np.unique(y)
        target_name = "label"

    if not images:
        raise ValueError(f"No images found in {root}")

    X = np.stack(images, axis=0).astype(np.float32)  # (n, H, W, C)
    return ImageDataset(
        X=X,
        y=y,
        feature_names=["image"],
        target_name=target_name,
        classes=classes,
        data_type="image",
    )


def _load_one(path: Path, config: ImageDatasetConfig) -> np.ndarray:
    """Load and normalise a single image to float32 (H, W, 3) in [0, 1]."""
    img = PILImage.open(path).convert("RGB")
    if config.image_size is not None:
        h, w = config.image_size
        img = img.resize((w, h), PILImage.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _validate_shapes(images: list[np.ndarray], root: Path) -> None:
    shapes = {img.shape for img in images}
    if len(shapes) > 1:
        raise ValueError(
            f"Inconsistent image shapes: {shapes}. "
            f"Set image_size in config to resize all images to the same shape."
        )


def _collect_labelled(
    root: Path, config: ImageDatasetConfig
) -> tuple[list[np.ndarray], list[str]]:
    images: list[np.ndarray] = []
    labels: list[str] = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in _EXTENSIONS:
                continue
            images.append(_load_one(img_path, config))
            labels.append(cls_dir.name)
    if config.image_size is None:
        _validate_shapes(images, root)
    return images, labels


def _collect_flat(root: Path, config: ImageDatasetConfig) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for img_path in sorted(root.iterdir()):
        if img_path.suffix.lower() not in _EXTENSIONS:
            continue
        images.append(_load_one(img_path, config))
    if config.image_size is None:
        _validate_shapes(images, root)
    return images
