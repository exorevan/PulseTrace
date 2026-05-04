"""Loader for text datasets stored as .txt files in class subdirectories."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pulsetrace.config.schema import TextDatasetConfig

from .text_dataset import TextDataset


def load_text_dataset(config: TextDatasetConfig) -> TextDataset:
    """Load a text classification dataset from a directory tree.

    Expected layout (only_x=False):
        path/
            class_a/  <- subdirectory name becomes label
                sample_001.txt
            class_b/
                sample_001.txt

    Expected layout (only_x=True):
        path/
            sample_001.txt  <- flat, no class subdirs
    """
    root = Path(config.path)
    if not root.exists():
        raise FileNotFoundError(f"Text dataset directory not found: {root}")

    if config.only_x:
        return _load_flat(root)
    return _load_subdirs(root)


def _load_flat(root: Path) -> TextDataset:
    """Load texts from a flat directory (no class subdirs, only_x mode)."""
    txt_files = sorted(root.glob("*.txt"))
    texts = [f.read_text(encoding="utf-8") for f in txt_files]
    if not texts:
        raise ValueError(f"No .txt files found in {root}")
    n = len(texts)
    return TextDataset(
        X=np.zeros((n, 1), dtype=np.float64),
        y=np.array([], dtype=object),
        feature_names=["text"],
        target_name="",
        classes=None,
        data_type="text",
        texts=texts,
    )


def _load_subdirs(root: Path) -> TextDataset:
    """Load texts from subdirectory-per-class layout."""
    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not subdirs:
        raise ValueError(f"No class subdirectories found in {root}")

    texts: list[str] = []
    labels: list[str] = []

    for subdir in subdirs:
        for txt_file in sorted(subdir.glob("*.txt")):
            texts.append(txt_file.read_text(encoding="utf-8"))
            labels.append(subdir.name)

    classes = [d.name for d in subdirs]

    n = len(texts)
    return TextDataset(
        X=np.zeros((n, 1), dtype=np.float64),
        y=np.array(labels, dtype=object),
        feature_names=["text"],
        target_name="label",
        classes=np.array(classes, dtype=object),
        data_type="text",
        texts=texts,
    )
