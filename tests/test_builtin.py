"""Tests for built-in dataset loading (sklearn + keras)."""
from __future__ import annotations

import numpy as np
import pytest

from pulsetrace.config.schema import BuiltinDatasetConfig
from pulsetrace.data import load_dataset
from pulsetrace.data.builtin_loader import AVAILABLE_NAMES, load_builtin
from pulsetrace.data.dataset import Dataset
from pulsetrace.data.image_dataset import ImageDataset


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

class TestBuiltinDatasetConfig:
    def test_minimal_config(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="iris")
        assert cfg.name == "iris"
        assert cfg.only_x is False
        assert cfg.split == "train"
        assert cfg.max_samples is None

    def test_all_fields(self):
        cfg = BuiltinDatasetConfig(
            type="builtin", name="mnist", split="test", max_samples=100, only_x=True
        )
        assert cfg.split == "test"
        assert cfg.max_samples == 100
        assert cfg.only_x is True


# ---------------------------------------------------------------------------
# Unknown name
# ---------------------------------------------------------------------------

class TestUnknownName:
    def test_raises_value_error(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="notadataset")
        with pytest.raises(ValueError, match="Unknown built-in dataset"):
            load_builtin(cfg)

    def test_error_lists_available_names(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="notadataset")
        with pytest.raises(ValueError, match="iris"):
            load_builtin(cfg)


# ---------------------------------------------------------------------------
# sklearn — classification datasets
# ---------------------------------------------------------------------------

class TestSklearnIris:
    def test_returns_dataset(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="iris")
        ds = load_builtin(cfg)
        assert isinstance(ds, Dataset)

    def test_shape(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="iris")
        ds = load_builtin(cfg)
        assert ds.X.shape == (150, 4)
        assert ds.y.shape == (150,)

    def test_feature_names(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="iris")
        ds = load_builtin(cfg)
        assert len(ds.feature_names) == 4
        assert all(isinstance(n, str) for n in ds.feature_names)

    def test_classes_not_none(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="iris")
        ds = load_builtin(cfg)
        assert ds.classes is not None
        assert len(ds.classes) == 3

    def test_y_contains_class_labels(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="iris")
        ds = load_builtin(cfg)
        assert set(ds.y).issubset(set(ds.classes))

    def test_x_dtype_float64(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="iris")
        ds = load_builtin(cfg)
        assert ds.X.dtype == np.float64

    def test_max_samples(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="iris", max_samples=20)
        ds = load_builtin(cfg)
        assert len(ds) == 20

    def test_only_x(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="iris", only_x=True)
        ds = load_builtin(cfg)
        assert len(ds.y) == 0
        assert ds.classes is None


class TestSklearnWine:
    def test_shape(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="wine")
        ds = load_builtin(cfg)
        assert ds.X.shape[1] == 13
        assert ds.classes is not None


class TestSklearnBreastCancer:
    def test_shape(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="breast_cancer")
        ds = load_builtin(cfg)
        assert ds.X.shape[1] == 30
        assert len(ds.classes) == 2


class TestSklearnDigits:
    def test_shape(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="digits")
        ds = load_builtin(cfg)
        assert ds.X.shape == (1797, 64)
        assert ds.classes is not None

    def test_feature_names_generated_when_missing(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="digits")
        ds = load_builtin(cfg)
        # digits may or may not have feature_names; either way we get valid strings
        assert len(ds.feature_names) == 64
        assert all(isinstance(n, str) for n in ds.feature_names)


# ---------------------------------------------------------------------------
# sklearn — regression datasets
# ---------------------------------------------------------------------------

class TestSklearnCaliforniaHousing:
    def test_returns_dataset(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="california_housing")
        ds = load_builtin(cfg)
        assert isinstance(ds, Dataset)

    def test_classes_none_for_regression(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="california_housing")
        ds = load_builtin(cfg)
        assert ds.classes is None

    def test_y_is_numeric(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="california_housing")
        ds = load_builtin(cfg)
        assert ds.y.shape[0] > 0


class TestSklearnDiabetes:
    def test_classes_none_for_regression(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="diabetes")
        ds = load_builtin(cfg)
        assert ds.classes is None

    def test_feature_names(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="diabetes")
        ds = load_builtin(cfg)
        assert len(ds.feature_names) == 10


# ---------------------------------------------------------------------------
# Routing via load_dataset
# ---------------------------------------------------------------------------

class TestLoadDatasetRouting:
    def test_routes_builtin_to_loader(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="iris")
        ds = load_dataset(cfg)
        assert isinstance(ds, Dataset)
        assert ds.X.shape == (150, 4)

    def test_available_names_includes_sklearn_and_keras(self):
        assert "iris" in AVAILABLE_NAMES
        assert "mnist" in AVAILABLE_NAMES
        assert "cifar10" in AVAILABLE_NAMES
        assert "fashion_mnist" in AVAILABLE_NAMES


# ---------------------------------------------------------------------------
# keras datasets (require download; marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestKerasMnist:
    def test_returns_image_dataset(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="mnist", max_samples=100)
        ds = load_builtin(cfg)
        assert isinstance(ds, ImageDataset)

    def test_shape_has_channel_dim(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="mnist", max_samples=50)
        ds = load_builtin(cfg)
        assert ds.X.shape == (50, 28, 28, 1)

    def test_normalized_range(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="mnist", max_samples=50)
        ds = load_builtin(cfg)
        assert ds.X.min() >= 0.0
        assert ds.X.max() <= 1.0

    def test_class_names_are_digit_strings(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="mnist", max_samples=10)
        ds = load_builtin(cfg)
        assert ds.classes is not None
        assert list(ds.classes) == [str(i) for i in range(10)]

    def test_split_test(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="mnist", split="test", max_samples=20)
        ds = load_builtin(cfg)
        assert len(ds) == 20

    def test_data_type_is_image(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="mnist", max_samples=10)
        ds = load_builtin(cfg)
        assert ds.data_type == "image"


@pytest.mark.slow
class TestKerasCifar10:
    def test_returns_image_dataset(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="cifar10", max_samples=50)
        ds = load_builtin(cfg)
        assert isinstance(ds, ImageDataset)

    def test_shape(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="cifar10", max_samples=50)
        ds = load_builtin(cfg)
        assert ds.X.shape == (50, 32, 32, 3)

    def test_class_names(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="cifar10", max_samples=10)
        ds = load_builtin(cfg)
        assert ds.classes is not None
        assert "airplane" in list(ds.classes)
        assert len(ds.classes) == 10


@pytest.mark.slow
class TestKerasFashionMnist:
    def test_returns_image_dataset(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="fashion_mnist", max_samples=50)
        ds = load_builtin(cfg)
        assert isinstance(ds, ImageDataset)

    def test_shape_has_channel_dim(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="fashion_mnist", max_samples=50)
        ds = load_builtin(cfg)
        assert ds.X.shape == (50, 28, 28, 1)

    def test_normalized_range(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="fashion_mnist", max_samples=50)
        ds = load_builtin(cfg)
        assert ds.X.min() >= 0.0
        assert ds.X.max() <= 1.0

    def test_class_names(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="fashion_mnist", max_samples=10)
        ds = load_builtin(cfg)
        assert ds.classes is not None
        assert "t-shirt" in list(ds.classes)
        assert "sneaker" in list(ds.classes)
        assert len(ds.classes) == 10

    def test_data_type_is_image(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="fashion_mnist", max_samples=10)
        ds = load_builtin(cfg)
        assert ds.data_type == "image"

    def test_split_test(self):
        cfg = BuiltinDatasetConfig(type="builtin", name="fashion_mnist", split="test", max_samples=20)
        ds = load_builtin(cfg)
        assert len(ds) == 20
