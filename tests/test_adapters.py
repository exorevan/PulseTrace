import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from pulsetrace.adapters.keras import KerasAdapter
from pulsetrace.adapters.pytorch import PyTorchAdapter
from pulsetrace.adapters.sklearn import SklearnAdapter


# ---- SklearnAdapter tests ----

@pytest.fixture(scope="module")
def iris_clf():
    X, y = load_iris(return_X_y=True)
    m = RandomForestClassifier(n_estimators=10, random_state=42)
    m.fit(X, y)
    return m, X


@pytest.fixture(scope="module")
def linear_reg():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    m = LinearRegression().fit(X, y)
    return m, X


def test_sklearn_clf_task(iris_clf):
    model, _ = iris_clf
    adapter = SklearnAdapter(model)
    assert adapter.task == "classification"


def test_sklearn_reg_task(linear_reg):
    model, _ = linear_reg
    adapter = SklearnAdapter(model)
    assert adapter.task == "regression"


def test_sklearn_predict_shape(iris_clf):
    model, X = iris_clf
    adapter = SklearnAdapter(model)
    preds = adapter.predict(X[:5])
    assert preds.shape == (5,)


def test_sklearn_predict_proba_shape(iris_clf):
    model, X = iris_clf
    adapter = SklearnAdapter(model)
    proba = adapter.predict_proba(X[:5])
    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-6)


def test_sklearn_regression_predict_proba_raises(linear_reg):
    model, X = linear_reg
    adapter = SklearnAdapter(model)
    with pytest.raises(NotImplementedError):
        adapter.predict_proba(X)


def test_sklearn_regression_predict_values(linear_reg):
    model, X = linear_reg
    adapter = SklearnAdapter(model)
    preds = adapter.predict(X)
    assert preds.shape == (4,)
    np.testing.assert_allclose(preds, [2.0, 4.0, 6.0, 8.0], atol=0.1)


# ---- KerasAdapter tests (mock model, no real Keras load) ----

class _MockKerasModel:
    """Minimal stand-in for a keras.Model."""

    def __init__(self, output_units: int) -> None:
        self.output_shape = (None, output_units)

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        n = len(X)
        return np.random.default_rng(0).random((n, self.output_shape[-1]))


def test_keras_regression_task():
    adapter = KerasAdapter(_MockKerasModel(output_units=1))
    assert adapter.task == "regression"


def test_keras_classification_task():
    adapter = KerasAdapter(_MockKerasModel(output_units=3))
    assert adapter.task == "classification"


def test_keras_regression_predict_shape():
    adapter = KerasAdapter(_MockKerasModel(output_units=1))
    preds = adapter.predict(np.random.rand(5, 4))
    assert preds.shape == (5,)


def test_keras_classification_predict_shape():
    adapter = KerasAdapter(_MockKerasModel(output_units=3))
    preds = adapter.predict(np.random.rand(5, 4))
    assert preds.shape == (5,)


def test_keras_predict_proba_shape():
    adapter = KerasAdapter(_MockKerasModel(output_units=3))
    proba = adapter.predict_proba(np.random.rand(5, 4))
    assert proba.shape == (5, 3)


def test_keras_regression_predict_proba_raises():
    adapter = KerasAdapter(_MockKerasModel(output_units=1))
    with pytest.raises(NotImplementedError):
        adapter.predict_proba(np.random.rand(3, 4))


# ---- PyTorchAdapter tests ----

class _SimpleTorchModel(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_pytorch_classification_task():
    model = _SimpleTorchModel(4, 3)
    adapter = PyTorchAdapter(model, task="classification")
    assert adapter.task == "classification"


def test_pytorch_regression_task():
    model = _SimpleTorchModel(4, 1)
    adapter = PyTorchAdapter(model, task="regression")
    assert adapter.task == "regression"


def test_pytorch_predict_classification_shape():
    model = _SimpleTorchModel(4, 3)
    adapter = PyTorchAdapter(model, task="classification")
    X = np.random.rand(5, 4).astype(np.float32)
    preds = adapter.predict(X)
    assert preds.shape == (5,)


def test_pytorch_predict_regression_shape():
    model = _SimpleTorchModel(4, 1)
    adapter = PyTorchAdapter(model, task="regression")
    X = np.random.rand(5, 4).astype(np.float32)
    preds = adapter.predict(X)
    assert preds.shape == (5,)


def test_pytorch_predict_proba_sums_to_one():
    model = _SimpleTorchModel(4, 3)
    adapter = PyTorchAdapter(model, task="classification")
    X = np.random.rand(5, 4).astype(np.float32)
    proba = adapter.predict_proba(X)
    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-5)


def test_pytorch_regression_predict_proba_raises():
    model = _SimpleTorchModel(4, 1)
    adapter = PyTorchAdapter(model, task="regression")
    with pytest.raises(NotImplementedError):
        adapter.predict_proba(np.random.rand(3, 4).astype(np.float32))


def test_as_frame_skips_dataframe_for_ndim_gt_2():
    """_as_frame must not attempt pd.DataFrame on N-D arrays (e.g. image batches)."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # Train on a DataFrame so the model stores feature_names_in_
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    model = LogisticRegression(max_iter=200).fit(df, [0, 1, 0])
    adapter = SklearnAdapter(model)
    assert adapter._feature_names == ["a", "b"]

    # 4D array (image batch) — must be returned unchanged, no DataFrame attempt
    X_4d = np.ones((2, 8, 8, 3), dtype=np.float32)
    result = adapter._as_frame(X_4d)
    assert result is X_4d


def test_build_adapter_sets_keras_backend_torch(monkeypatch, tmp_path):
    """build_adapter must set KERAS_BACKEND='torch' before importing keras."""
    import os
    import sys
    from unittest.mock import MagicMock, patch

    from pulsetrace.adapters import build_adapter
    from pulsetrace.config.schema import KerasModelConfig

    # Remove any pre-existing KERAS_BACKEND so setdefault actually fires
    monkeypatch.delitem(os.environ, "KERAS_BACKEND", raising=False)

    mock_model = MagicMock()
    mock_model.output_shape = (None, 1)

    # Create a mock keras.models.load_model
    mock_load_model = MagicMock(return_value=mock_model)
    mock_models = MagicMock(load_model=mock_load_model)
    mock_keras = MagicMock(models=mock_models)

    cfg = KerasModelConfig(type="keras", path=str(tmp_path / "dummy.keras"))

    # Patch keras at the module level during import
    with patch.dict(sys.modules, {"keras": mock_keras, "keras.models": mock_models}):
        build_adapter(cfg)

    assert os.environ.get("KERAS_BACKEND") == "torch"
