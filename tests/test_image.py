"""Tests for image data loading, config, and end-to-end explanations."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from pulsetrace.config.schema import ImageDatasetConfig


class TestImageDatasetConfig:
    def test_parses_minimal(self, tmp_path):
        cfg = ImageDatasetConfig(type="image", path=tmp_path)
        assert cfg.path == tmp_path
        assert cfg.image_size is None
        assert cfg.only_x is False

    def test_parses_with_image_size(self, tmp_path):
        cfg = ImageDatasetConfig(type="image", path=tmp_path, image_size=[224, 224])
        assert cfg.image_size == [224, 224]

    def test_image_size_one_element_raises(self, tmp_path):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="image_size must have exactly 2 elements"):
            ImageDatasetConfig(type="image", path=tmp_path, image_size=[224])

    def test_image_size_three_elements_raises(self, tmp_path):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="image_size must have exactly 2 elements"):
            ImageDatasetConfig(type="image", path=tmp_path, image_size=[224, 224, 3])


from pulsetrace.data.image_dataset import ImageDataset


class TestImageDataset:
    def test_image_shape_rgb(self):
        X = np.zeros((5, 8, 8, 3), dtype=np.float32)
        ds = ImageDataset(
            X=X, y=np.array([], dtype=object),
            feature_names=["image"], target_name="",
            classes=None, data_type="image",
        )
        assert ds.image_shape == (8, 8, 3)

    def test_image_shape_grayscale(self):
        X = np.zeros((3, 16, 16, 1), dtype=np.float32)
        ds = ImageDataset(
            X=X, y=np.array([], dtype=object),
            feature_names=["image"], target_name="",
            classes=None, data_type="image",
        )
        assert ds.image_shape == (16, 16, 1)

    def test_instance_returns_3d(self):
        X = np.ones((4, 8, 8, 3), dtype=np.float32)
        ds = ImageDataset(
            X=X, y=np.array([], dtype=object),
            feature_names=["image"], target_name="",
            classes=None, data_type="image",
        )
        assert ds.instance(0).shape == (8, 8, 3)

    def test_is_subclass_of_dataset(self):
        from pulsetrace.data.dataset import Dataset
        X = np.zeros((2, 4, 4, 3), dtype=np.float32)
        ds = ImageDataset(
            X=X, y=np.array([], dtype=object),
            feature_names=["image"], target_name="",
            classes=None, data_type="image",
        )
        assert isinstance(ds, Dataset)


from PIL import Image as PILImage
from pulsetrace.config.schema import ImageDatasetConfig
from pulsetrace.data.image_loader import load_image_dataset
from pulsetrace.data import load_dataset


def _make_image_dir(root, classes=("cat", "dog"), n_per_class=4, size=(8, 8)):
    """Helper: create a synthetic subdirectory-labelled image directory."""
    rng = np.random.RandomState(42)
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = (rng.rand(*size, 3) * 255).astype(np.uint8)
            PILImage.fromarray(arr, mode="RGB").save(d / f"{i}.jpg")
    return root


class TestImageLoader:
    def test_subdirectory_labels_shape(self, tmp_path):
        img_dir = _make_image_dir(tmp_path)
        cfg = ImageDatasetConfig(type="image", path=img_dir, image_size=[8, 8])
        ds = load_image_dataset(cfg)
        assert isinstance(ds, ImageDataset)
        assert ds.X.shape == (8, 8, 8, 3)   # 4+4 images, 8×8 RGB
        assert ds.y.shape == (8,)
        assert set(ds.y) == {"cat", "dog"}

    def test_only_x_flat_dir(self, tmp_path):
        flat_dir = tmp_path / "flat"
        flat_dir.mkdir()
        rng = np.random.RandomState(0)
        for i in range(3):
            arr = (rng.rand(8, 8, 3) * 255).astype(np.uint8)
            PILImage.fromarray(arr, mode="RGB").save(flat_dir / f"{i}.jpg")
        cfg = ImageDatasetConfig(type="image", path=flat_dir, image_size=[8, 8], only_x=True)
        ds = load_image_dataset(cfg)
        assert ds.X.shape == (3, 8, 8, 3)
        assert len(ds.y) == 0

    def test_inconsistent_sizes_raises(self, tmp_path):
        d = tmp_path / "cls"
        d.mkdir()
        PILImage.new("RGB", (8, 8)).save(d / "a.jpg")
        PILImage.new("RGB", (16, 16)).save(d / "b.jpg")
        cfg = ImageDatasetConfig(type="image", path=tmp_path)  # no image_size
        with pytest.raises(ValueError, match="Inconsistent image shapes"):
            load_image_dataset(cfg)

    def test_resize_normalises_to_target_size(self, tmp_path):
        d = tmp_path / "cls"
        d.mkdir()
        PILImage.new("RGB", (32, 32)).save(d / "big.jpg")
        PILImage.new("RGB", (64, 48)).save(d / "bigger.jpg")
        cfg = ImageDatasetConfig(type="image", path=tmp_path, image_size=[8, 8])
        ds = load_image_dataset(cfg)
        assert ds.X.shape == (2, 8, 8, 3)

    def test_data_type_is_image(self, tmp_path):
        img_dir = _make_image_dir(tmp_path)
        cfg = ImageDatasetConfig(type="image", path=img_dir, image_size=[8, 8])
        ds = load_image_dataset(cfg)
        assert ds.data_type == "image"

    def test_pixel_values_normalised_0_to_1(self, tmp_path):
        d = tmp_path / "cls"
        d.mkdir()
        arr = np.full((8, 8, 3), 255, dtype=np.uint8)
        PILImage.fromarray(arr).save(d / "white.jpg")
        cfg = ImageDatasetConfig(type="image", path=tmp_path, image_size=[8, 8])
        ds = load_image_dataset(cfg)
        assert ds.X.max() <= 1.0
        assert ds.X.min() >= 0.0

    def test_routes_via_load_dataset(self, tmp_path):
        img_dir = _make_image_dir(tmp_path)
        cfg = ImageDatasetConfig(type="image", path=img_dir, image_size=[8, 8])
        ds = load_dataset(cfg)
        assert isinstance(ds, ImageDataset)
        assert ds.data_type == "image"


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from pulsetrace.adapters.sklearn import SklearnAdapter
from pulsetrace.config.schema import ExplainerConfig
from pulsetrace.explainers.lime import LimeExplainer
from pulsetrace.explainers.result import ExplanationResult


@pytest.fixture(scope="module")
def img_classification_adapter_and_dataset():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 8, 8, 3).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10)
    dataset = ImageDataset(
        X=X, y=y.astype(object),
        feature_names=["image"], target_name="label",
        classes=np.array([0, 1], dtype=object), data_type="image",
    )
    model = Pipeline([
        ("flatten", FunctionTransformer(lambda x: x.reshape(len(x), -1), validate=False)),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    model.fit(X.reshape(20, -1), y)
    return SklearnAdapter(model), dataset


@pytest.fixture(scope="module")
def img_regression_adapter_and_dataset():
    rng = np.random.RandomState(1)
    X = rng.rand(20, 8, 8, 3).astype(np.float32)
    y = X[:, 0, 0, 0] * 3.0 + X[:, 4, 4, 2] * 1.5
    dataset = ImageDataset(
        X=X, y=y.astype(object),
        feature_names=["image"], target_name="value",
        classes=None, data_type="image",
    )
    model = Pipeline([
        ("flatten", FunctionTransformer(lambda x: x.reshape(len(x), -1), validate=False)),
        ("reg", LinearRegression()),
    ])
    model.fit(X.reshape(20, -1), y)
    return SklearnAdapter(model), dataset


_LIME_IMG_CFG = ExplainerConfig(type="lime", num_features=3, num_samples=50, n_segments=6)


class TestLimeImageGlobal:
    def test_returns_explanation_result(self, img_classification_adapter_and_dataset):
        adapter, dataset = img_classification_adapter_and_dataset
        result = LimeExplainer(_LIME_IMG_CFG).explain_global(adapter, dataset)
        assert isinstance(result, ExplanationResult)

    def test_mode_is_global(self, img_classification_adapter_and_dataset):
        adapter, dataset = img_classification_adapter_and_dataset
        result = LimeExplainer(_LIME_IMG_CFG).explain_global(adapter, dataset)
        assert result.mode == "global"

    def test_contributions_use_region_names(self, img_classification_adapter_and_dataset):
        adapter, dataset = img_classification_adapter_and_dataset
        result = LimeExplainer(_LIME_IMG_CFG).explain_global(adapter, dataset)
        assert all(c.feature.startswith("region_") for c in result.contributions)

    def test_regression_has_base_value(self, img_regression_adapter_and_dataset):
        adapter, dataset = img_regression_adapter_and_dataset
        result = LimeExplainer(_LIME_IMG_CFG).explain_global(adapter, dataset)
        assert result.base_values is not None
        assert None in result.base_values


class TestLimeImageLocal:
    def test_local_classification(self, img_classification_adapter_and_dataset):
        adapter, dataset = img_classification_adapter_and_dataset
        instance = ImageDataset(
            X=dataset.X[:1], y=np.array([], dtype=object),
            feature_names=["image"], target_name="",
            classes=None, data_type="image",
        )
        result = LimeExplainer(_LIME_IMG_CFG).explain_local(adapter, instance, dataset)
        assert result.mode == "local"
        assert all(c.feature.startswith("region_") for c in result.contributions)

    def test_local_regression(self, img_regression_adapter_and_dataset):
        adapter, dataset = img_regression_adapter_and_dataset
        instance = ImageDataset(
            X=dataset.X[:1], y=np.array([], dtype=object),
            feature_names=["image"], target_name="",
            classes=None, data_type="image",
        )
        result = LimeExplainer(_LIME_IMG_CFG).explain_local(adapter, instance, dataset)
        assert result.mode == "local"
        assert result.base_values is not None


from pulsetrace.explainers.shap import ShapExplainer

_SHAP_IMG_CFG = ExplainerConfig(type="shap", num_features=3)


class TestShapImageGlobal:
    def test_returns_explanation_result(self, img_classification_adapter_and_dataset):
        adapter, dataset = img_classification_adapter_and_dataset
        result = ShapExplainer(_SHAP_IMG_CFG).explain_global(adapter, dataset)
        assert isinstance(result, ExplanationResult)

    def test_mode_is_global(self, img_classification_adapter_and_dataset):
        adapter, dataset = img_classification_adapter_and_dataset
        result = ShapExplainer(_SHAP_IMG_CFG).explain_global(adapter, dataset)
        assert result.mode == "global"

    def test_contributions_use_region_names(self, img_classification_adapter_and_dataset):
        adapter, dataset = img_classification_adapter_and_dataset
        result = ShapExplainer(_SHAP_IMG_CFG).explain_global(adapter, dataset)
        assert all(c.feature.startswith("region_") for c in result.contributions)

    def test_at_most_num_features_per_class(self, img_classification_adapter_and_dataset):
        adapter, dataset = img_classification_adapter_and_dataset
        result = ShapExplainer(_SHAP_IMG_CFG).explain_global(adapter, dataset)
        # contributions covers all classes, each class gets ≤ num_features
        assert len(result.contributions) <= _SHAP_IMG_CFG.num_features * len(dataset.classes)

    def test_regression_has_base_value(self, img_regression_adapter_and_dataset):
        adapter, dataset = img_regression_adapter_and_dataset
        result = ShapExplainer(_SHAP_IMG_CFG).explain_global(adapter, dataset)
        assert result.base_values is not None
        assert None in result.base_values


class TestShapImageLocal:
    def test_local_classification(self, img_classification_adapter_and_dataset):
        adapter, dataset = img_classification_adapter_and_dataset
        instance = ImageDataset(
            X=dataset.X[:1], y=np.array([], dtype=object),
            feature_names=["image"], target_name="",
            classes=None, data_type="image",
        )
        result = ShapExplainer(_SHAP_IMG_CFG).explain_local(adapter, instance, dataset)
        assert result.mode == "local"
        assert all(c.feature.startswith("region_") for c in result.contributions)

    def test_local_regression(self, img_regression_adapter_and_dataset):
        adapter, dataset = img_regression_adapter_and_dataset
        instance = ImageDataset(
            X=dataset.X[:1], y=np.array([], dtype=object),
            feature_names=["image"], target_name="",
            classes=None, data_type="image",
        )
        result = ShapExplainer(_SHAP_IMG_CFG).explain_local(adapter, instance, dataset)
        assert result.mode == "local"
        assert result.base_values is not None


import yaml
import joblib
from pulsetrace.main import run


def _flatten_images(x):
    """Module-level flatten function (picklable by joblib)."""
    return x.reshape(len(x), -1)


class TestImageEndToEnd:
    def test_shap_global_classification_pipeline(self, tmp_path, capsys):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.linear_model import LogisticRegression

        # Create synthetic image directory (2 classes, 4 images each, 8×8 RGB)
        rng = np.random.RandomState(7)
        for cls in ("cat", "dog"):
            d = tmp_path / "images" / cls
            d.mkdir(parents=True)
            for i in range(4):
                arr = (rng.rand(8, 8, 3) * 255).astype(np.uint8)
                PILImage.fromarray(arr, mode="RGB").save(d / f"{i}.jpg")

        # Build and save a flattening sklearn pipeline
        X_flat = rng.rand(8, 8 * 8 * 3).astype(np.float32)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        model = Pipeline([
            ("flatten", FunctionTransformer(_flatten_images, validate=False)),
            ("clf", LogisticRegression(max_iter=200)),
        ])
        model.fit(X_flat, y)
        model_path = tmp_path / "model.pkl"
        joblib.dump(model, model_path)

        # Write YAML config
        cfg = {
            "model": {"type": "sklearn", "path": str(model_path)},
            "dataset": {"type": "image", "path": str(tmp_path / "images"), "image_size": [8, 8]},
            "explainer": {"type": "shap", "num_features": 3},
        }
        cfg_path = tmp_path / "img_cfg.yaml"
        cfg_path.write_text(yaml.dump(cfg))

        capsys.readouterr()
        run(cfg_path)
        out = capsys.readouterr().out

        assert "SHAP" in out
        assert "mode=global" in out
        assert "task=classification" in out
