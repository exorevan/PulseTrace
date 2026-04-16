"""Load built-in datasets from sklearn and keras without local files."""
from __future__ import annotations

import numpy as np

from pulsetrace.config.schema import BuiltinDatasetConfig

from .dataset import Dataset
from .image_dataset import ImageDataset

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SKLEARN_DATASETS: dict[str, str] = {
    "iris": "load_iris",
    "wine": "load_wine",
    "breast_cancer": "load_breast_cancer",
    "digits": "load_digits",
    "california_housing": "fetch_california_housing",
    "diabetes": "load_diabetes",
}

_KERAS_DATASETS: dict[str, tuple[list[str], str]] = {
    "mnist": (
        [str(i) for i in range(10)],
        "target",
    ),
    "cifar10": (
        ["airplane", "automobile", "bird", "cat", "deer",
         "dog", "frog", "horse", "ship", "truck"],
        "target",
    ),
    "fashion_mnist": (
        ["t-shirt", "trouser", "pullover", "dress", "coat",
         "sandal", "shirt", "sneaker", "bag", "ankle_boot"],
        "target",
    ),
}

AVAILABLE_NAMES: list[str] = sorted(_SKLEARN_DATASETS) + sorted(_KERAS_DATASETS)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_builtin(config: BuiltinDatasetConfig) -> Dataset | ImageDataset:
    name = config.name.lower()
    if name in _SKLEARN_DATASETS:
        return _load_sklearn(config)
    if name in _KERAS_DATASETS:
        return _load_keras(config)
    raise ValueError(
        f"Unknown built-in dataset '{config.name}'. "
        f"Available: {', '.join(AVAILABLE_NAMES)}"
    )


# ---------------------------------------------------------------------------
# sklearn backend
# ---------------------------------------------------------------------------

def _load_sklearn(config: BuiltinDatasetConfig) -> Dataset:
    import sklearn.datasets as skds

    loader_name = _SKLEARN_DATASETS[config.name.lower()]
    bunch = getattr(skds, loader_name)()

    X: np.ndarray = bunch.data.astype(np.float64)
    target: np.ndarray = bunch.target

    raw_names = getattr(bunch, "feature_names", None)
    if raw_names is not None and len(raw_names) > 0:
        feature_names: list[str] = [str(n) for n in raw_names]
    else:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    # Regression targets are float; classification targets are integer-valued
    is_classification = np.issubdtype(bunch.target.dtype, np.integer)
    target_name = getattr(bunch, "target_name", "target")
    if isinstance(target_name, list):
        target_name = target_name[0] if target_name else "target"

    if config.max_samples is not None:
        X = X[: config.max_samples]
        target = target[: config.max_samples]

    if config.only_x:
        return Dataset(
            X=X,
            y=np.array([], dtype=object),
            feature_names=feature_names,
            target_name=target_name,
            classes=None,
        )

    if is_classification:
        class_labels = np.array(bunch.target_names, dtype=object)
        y = np.array([class_labels[int(t)] for t in target], dtype=object)
        classes = class_labels
    else:
        y = target.astype(object)
        classes = None

    return Dataset(
        X=X,
        y=y,
        feature_names=feature_names,
        target_name=target_name,
        classes=classes,
    )


# ---------------------------------------------------------------------------
# keras backend
# ---------------------------------------------------------------------------

def _load_keras(config: BuiltinDatasetConfig) -> ImageDataset:
    import importlib
    name = config.name.lower()
    mod = importlib.import_module(f"keras.datasets.{name}")
    (x_train, y_train), (x_test, y_test) = mod.load_data()

    x = x_train if config.split == "train" else x_test
    y = y_train if config.split == "train" else y_test

    # Ensure channel dimension: (n, H, W) → (n, H, W, 1)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=-1)

    # Normalize to float64 [0, 1]
    x = x.astype(np.float64) / 255.0

    y = y.ravel()

    if config.max_samples is not None:
        x = x[: config.max_samples]
        y = y[: config.max_samples]

    class_names, _ = _KERAS_DATASETS[name]
    class_labels = np.array(class_names, dtype=object)

    if config.only_x:
        return ImageDataset(
            X=x,
            y=np.array([], dtype=object),
            feature_names=[],
            target_name="label",
            classes=None,
            data_type="image",
        )

    y_named = np.array([class_names[int(t)] for t in y], dtype=object)

    return ImageDataset(
        X=x,
        y=y_named,
        feature_names=[],
        target_name="label",
        classes=class_labels,
        data_type="image",
    )
