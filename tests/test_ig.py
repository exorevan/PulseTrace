from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class _TinyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture()
def tabular_clf_adapter():
    from pulsetrace.adapters.pytorch import PyTorchAdapter
    model = _TinyLinear(4, 2)
    return PyTorchAdapter(model, task="classification")


@pytest.fixture()
def tabular_reg_adapter():
    from pulsetrace.adapters.pytorch import PyTorchAdapter
    model = _TinyLinear(4, 1)
    return PyTorchAdapter(model, task="regression")


@pytest.fixture()
def tabular_dataset():
    from pulsetrace.data.dataset import Dataset
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = np.array([0, 1] * 10, dtype=object)
    return Dataset(
        X=X, y=y,
        feature_names=["f0", "f1", "f2", "f3"],
        target_name="label",
        classes=np.array([0, 1], dtype=object),
    )


# ---------------------------------------------------------------------------
# Task 1: Config
# ---------------------------------------------------------------------------

def test_ig_config_type_accepted():
    from pulsetrace.config.schema import ExplainerConfig
    cfg = ExplainerConfig(type="ig")
    assert cfg.type == "ig"


def test_ig_config_defaults():
    from pulsetrace.config.schema import ExplainerConfig
    cfg = ExplainerConfig(type="ig")
    assert cfg.ig_steps == 50
    assert cfg.ig_baseline == "zero"


def test_ig_config_custom_values():
    from pulsetrace.config.schema import ExplainerConfig
    cfg = ExplainerConfig(type="ig", ig_steps=20, ig_baseline="mean")
    assert cfg.ig_steps == 20
    assert cfg.ig_baseline == "mean"


# ---------------------------------------------------------------------------
# Task 2: Backend detection
# ---------------------------------------------------------------------------

def test_ig_rejects_sklearn_adapter():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris

    from pulsetrace.adapters.sklearn import SklearnAdapter
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.explainers.ig import IgExplainer

    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    adapter = SklearnAdapter(clf)

    from pulsetrace.data.dataset import Dataset
    ds = Dataset(X=X[:10].astype(np.float32), y=y[:10].astype(object),
                 feature_names=[f"f{i}" for i in range(4)], target_name="t",
                 classes=np.array([0, 1, 2], dtype=object))

    cfg = ExplainerConfig(type="ig", ig_steps=5)
    explainer = IgExplainer(cfg)
    with pytest.raises(NotImplementedError, match="pt or hf"):
        explainer.explain_local(adapter, ds, ds)


def test_ig_rejects_keras_adapter():
    from unittest.mock import MagicMock

    from pulsetrace.adapters.keras import KerasAdapter
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.data.dataset import Dataset
    from pulsetrace.explainers.ig import IgExplainer

    mock_model = MagicMock()
    mock_model.output_shape = (None, 2)
    adapter = KerasAdapter(mock_model)

    X = np.zeros((5, 4), dtype=np.float32)
    ds = Dataset(
        X=X, y=np.array([0] * 5, dtype=object),
        feature_names=["f0", "f1", "f2", "f3"], target_name="t",
        classes=np.array([0, 1], dtype=object),
    )
    cfg = ExplainerConfig(type="ig", ig_steps=5)
    explainer = IgExplainer(cfg)
    with pytest.raises(NotImplementedError, match="Keras support is planned"):
        explainer.explain_local(adapter, ds, ds)


# ---------------------------------------------------------------------------
# Task 3: Registration
# ---------------------------------------------------------------------------

def test_build_explainer_returns_ig_explainer():
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.explainers import build_explainer
    from pulsetrace.explainers.ig import IgExplainer

    cfg = ExplainerConfig(type="ig")
    explainer = build_explainer(cfg)
    assert isinstance(explainer, IgExplainer)


# ---------------------------------------------------------------------------
# Task 4: Tabular local + global
# ---------------------------------------------------------------------------

def test_ig_tabular_local_classification(tabular_clf_adapter, tabular_dataset):
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.explainers.ig import IgExplainer
    from pulsetrace.data.dataset import Dataset

    cfg = ExplainerConfig(type="ig", num_features=4, ig_steps=5)
    explainer = IgExplainer(cfg)
    instance = Dataset(
        X=tabular_dataset.X[:1], y=tabular_dataset.y[:1],
        feature_names=tabular_dataset.feature_names,
        target_name=tabular_dataset.target_name,
        classes=tabular_dataset.classes,
    )
    result = explainer.explain_local(tabular_clf_adapter, instance, tabular_dataset)

    assert result.mode == "local"
    assert result.method == "ig"
    assert result.task == "classification"
    assert len(result.contributions) == 4
    assert all(fc.label is not None for fc in result.contributions)
    assert result.base_values is not None


def test_ig_tabular_local_regression(tabular_reg_adapter, tabular_dataset):
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.explainers.ig import IgExplainer
    from pulsetrace.data.dataset import Dataset

    reg_dataset = Dataset(
        X=tabular_dataset.X, y=np.zeros(len(tabular_dataset), dtype=object),
        feature_names=tabular_dataset.feature_names,
        target_name="value", classes=None,
    )
    cfg = ExplainerConfig(type="ig", num_features=4, ig_steps=5)
    explainer = IgExplainer(cfg)
    instance = Dataset(
        X=tabular_dataset.X[:1], y=np.array([0.0], dtype=object),
        feature_names=tabular_dataset.feature_names,
        target_name="value", classes=None,
    )
    result = explainer.explain_local(tabular_reg_adapter, instance, reg_dataset)

    assert result.mode == "local"
    assert result.task == "regression"
    assert all(fc.label is None for fc in result.contributions)


def test_ig_tabular_global_classification(tabular_clf_adapter, tabular_dataset):
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.explainers.ig import IgExplainer

    cfg = ExplainerConfig(type="ig", num_features=4, ig_steps=5, global_samples=3)
    explainer = IgExplainer(cfg)
    result = explainer.explain_global(tabular_clf_adapter, tabular_dataset)

    assert result.mode == "global"
    assert result.method == "ig"
    assert result.global_samples == 3
    assert len(result.contributions) > 0


def test_ig_tabular_local_mean_baseline(tabular_clf_adapter, tabular_dataset):
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.explainers.ig import IgExplainer
    from pulsetrace.data.dataset import Dataset

    cfg = ExplainerConfig(type="ig", num_features=4, ig_steps=5, ig_baseline="mean")
    explainer = IgExplainer(cfg)
    instance = Dataset(
        X=tabular_dataset.X[:1], y=tabular_dataset.y[:1],
        feature_names=tabular_dataset.feature_names,
        target_name=tabular_dataset.target_name,
        classes=tabular_dataset.classes,
    )
    result = explainer.explain_local(tabular_clf_adapter, instance, tabular_dataset)
    assert result.mode == "local"
    assert len(result.contributions) == 4


# ---------------------------------------------------------------------------
# Task 5: Image local + global
# ---------------------------------------------------------------------------

@pytest.fixture()
def image_clf_adapter():
    from pulsetrace.adapters.pytorch import PyTorchAdapter

    class _FlatConv(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8 * 8 * 1, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x.reshape(x.shape[0], -1))

    return PyTorchAdapter(_FlatConv(), task="classification")


@pytest.fixture()
def image_dataset():
    from pulsetrace.data.image_dataset import ImageDataset

    rng = np.random.default_rng(1)
    X = rng.random((10, 8, 8, 1)).astype(np.float32)
    y = np.array([0, 1] * 5, dtype=object)
    return ImageDataset(
        X=X, y=y,
        feature_names=[f"px_{i}" for i in range(64)],
        target_name="cls",
        classes=np.array([0, 1], dtype=object),
        data_type="image",
    )


def test_ig_image_local_returns_panel(image_clf_adapter, image_dataset):
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.explainers.ig import IgExplainer
    from pulsetrace.data.image_dataset import ImageDataset

    cfg = ExplainerConfig(type="ig", num_features=10, ig_steps=5)
    explainer = IgExplainer(cfg)
    instance = ImageDataset(
        X=image_dataset.X[:1], y=image_dataset.y[:1],
        feature_names=image_dataset.feature_names,
        target_name=image_dataset.target_name,
        classes=image_dataset.classes,
        data_type="image",
    )
    result = explainer.explain_local(image_clf_adapter, instance, image_dataset)

    assert result.mode == "local"
    assert result.method == "ig"
    assert result.image_panels is not None
    assert len(result.image_panels) == 1


def test_ig_image_global_returns_panels(image_clf_adapter, image_dataset):
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.explainers.ig import IgExplainer

    cfg = ExplainerConfig(type="ig", num_features=10, ig_steps=5, global_samples=3)
    explainer = IgExplainer(cfg)
    result = explainer.explain_global(image_clf_adapter, image_dataset)

    assert result.mode == "global"
    assert result.image_panels is not None
    assert len(result.image_panels) == 3


# ---------------------------------------------------------------------------
# Task 6: Timeseries local + global
# ---------------------------------------------------------------------------

@pytest.fixture()
def ts_clf_adapter():
    from pulsetrace.adapters.pytorch import PyTorchAdapter

    class _TsLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(20, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    return PyTorchAdapter(_TsLinear(), task="classification")


@pytest.fixture()
def ts_dataset():
    from pulsetrace.data.timeseries_dataset import TimeSeriesDataset

    rng = np.random.default_rng(2)
    X = rng.standard_normal((15, 20)).astype(np.float32)
    y = np.array([0, 1] * 7 + [0], dtype=object)
    return TimeSeriesDataset(
        X=X, y=y,
        feature_names=[f"t{i}" for i in range(20)],
        target_name="cls",
        classes=np.array([0, 1], dtype=object),
        data_type="timeseries",
    )


def test_ig_ts_local_returns_panel(ts_clf_adapter, ts_dataset):
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.explainers.ig import IgExplainer
    from pulsetrace.data.timeseries_dataset import TimeSeriesDataset

    cfg = ExplainerConfig(type="ig", num_features=20, ig_steps=5)
    explainer = IgExplainer(cfg)
    instance = TimeSeriesDataset(
        X=ts_dataset.X[:1], y=ts_dataset.y[:1],
        feature_names=ts_dataset.feature_names,
        target_name=ts_dataset.target_name,
        classes=ts_dataset.classes,
        data_type="timeseries",
    )
    result = explainer.explain_local(ts_clf_adapter, instance, ts_dataset)

    assert result.mode == "local"
    assert result.method == "ig"
    assert result.image_panels is not None
    assert len(result.image_panels) == 1


def test_ig_ts_global_returns_panels(ts_clf_adapter, ts_dataset):
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.explainers.ig import IgExplainer

    cfg = ExplainerConfig(type="ig", num_features=20, ig_steps=5, global_samples=3)
    explainer = IgExplainer(cfg)
    result = explainer.explain_global(ts_clf_adapter, ts_dataset)

    assert result.mode == "global"
    assert result.image_panels is not None
    assert len(result.image_panels) == 3


# ---------------------------------------------------------------------------
# Task 7: Text local + global (HuggingFace)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hf_adapter():
    from pulsetrace.adapters.huggingface import HfAdapter
    from pulsetrace.config.schema import HfModelConfig

    cfg = HfModelConfig(
        type="hf",
        path_or_name="distilbert-base-uncased-finetuned-sst-2-english",
        labels=["negative", "positive"],
        max_length=64,
    )
    return HfAdapter(cfg)


@pytest.fixture(scope="module")
def text_dataset():
    from pulsetrace.data.text_dataset import TextDataset

    texts = [
        "I love this film",
        "This movie is terrible",
        "Great acting and story",
    ]
    y = np.array(["positive", "negative", "positive"], dtype=object)
    return TextDataset(
        X=np.zeros((3, 1), dtype=np.float32),
        y=y,
        feature_names=["text"],
        target_name="sentiment",
        classes=np.array(["negative", "positive"], dtype=object),
        data_type="text",
        texts=texts,
    )


@pytest.mark.slow
def test_ig_text_local(hf_adapter, text_dataset):
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.data.text_dataset import TextDataset
    from pulsetrace.explainers.ig import IgExplainer

    cfg = ExplainerConfig(type="ig", num_features=5, ig_steps=5)
    explainer = IgExplainer(cfg)

    instance = TextDataset(
        X=np.zeros((1, 1), dtype=np.float32),
        y=np.array(["positive"], dtype=object),
        feature_names=["text"],
        target_name="sentiment",
        classes=text_dataset.classes,
        data_type="text",
        texts=["I love this film"],
    )
    result = explainer.explain_local(hf_adapter, instance, text_dataset)

    assert result.mode == "local"
    assert result.method == "ig"
    assert result.image_panels is not None
    assert len(result.image_panels) == 1
    assert len(result.contributions) > 0


@pytest.mark.slow
def test_ig_text_global(hf_adapter, text_dataset):
    from pulsetrace.config.schema import ExplainerConfig
    from pulsetrace.explainers.ig import IgExplainer

    cfg = ExplainerConfig(type="ig", num_features=5, ig_steps=5, global_samples=2)
    explainer = IgExplainer(cfg)
    result = explainer.explain_global(hf_adapter, text_dataset)

    assert result.mode == "global"
    assert result.image_panels is not None
    assert len(result.image_panels) == 2
