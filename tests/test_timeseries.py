"""Tests for time series data loading, config, and end-to-end explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from pulsetrace.config.schema import TimeSeriesDatasetConfig, ExplainerConfig
from pulsetrace.data.dataset import Dataset
from pulsetrace.data.timeseries_dataset import TimeSeriesDataset
from pulsetrace.data.timeseries_loader import load_timeseries


# ---------------------------------------------------------------------------
# Loader — CSV
# ---------------------------------------------------------------------------

@pytest.fixture
def ts_csv_with_target(tmp_path):
    """CSV where last column is the class label."""
    data = {f"t{i}": np.random.rand(10) for i in range(20)}
    data["label"] = ["A"] * 5 + ["B"] * 5
    df = pd.DataFrame(data)
    path = tmp_path / "ts.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def ts_csv_only_x(tmp_path):
    """CSV with no label column."""
    data = {f"t{i}": np.random.rand(5) for i in range(15)}
    df = pd.DataFrame(data)
    path = tmp_path / "ts_x.csv"
    df.to_csv(path, index=False)
    return path


class TestLoadTimeseriesCSV:
    def test_shape_with_target(self, ts_csv_with_target):
        cfg = TimeSeriesDatasetConfig(type="timeseries", path=ts_csv_with_target, target_col="label")
        ds = load_timeseries(cfg)
        assert ds.X.shape == (10, 20)
        assert ds.y.shape == (10,)

    def test_feature_names_are_t0_tN(self, ts_csv_with_target):
        cfg = TimeSeriesDatasetConfig(type="timeseries", path=ts_csv_with_target, target_col="label")
        ds = load_timeseries(cfg)
        assert ds.feature_names == [f"t{i}" for i in range(20)]

    def test_data_type_is_timeseries(self, ts_csv_with_target):
        cfg = TimeSeriesDatasetConfig(type="timeseries", path=ts_csv_with_target, target_col="label")
        ds = load_timeseries(cfg)
        assert ds.data_type == "timeseries"

    def test_only_x_has_empty_y(self, ts_csv_only_x):
        cfg = TimeSeriesDatasetConfig(type="timeseries", path=ts_csv_only_x, only_x=True)
        ds = load_timeseries(cfg)
        assert ds.X.shape == (5, 15)
        assert len(ds.y) == 0

    def test_n_timesteps_validation_passes(self, ts_csv_with_target):
        cfg = TimeSeriesDatasetConfig(type="timeseries", path=ts_csv_with_target, target_col="label", n_timesteps=20)
        ds = load_timeseries(cfg)
        assert ds.X.shape[1] == 20

    def test_n_timesteps_validation_fails(self, ts_csv_with_target):
        cfg = TimeSeriesDatasetConfig(type="timeseries", path=ts_csv_with_target, target_col="label", n_timesteps=99)
        with pytest.raises(ValueError, match="Expected 99 timesteps"):
            load_timeseries(cfg)


# ---------------------------------------------------------------------------
# Loader — NumPy
# ---------------------------------------------------------------------------

@pytest.fixture
def ts_npy_with_target(tmp_path):
    """Npy array shape (20, 11) — first 10 cols are features, last is target."""
    arr = np.random.rand(20, 11)
    arr[:, -1] = np.where(arr[:, -1] > 0.5, 1.0, 0.0)
    path = tmp_path / "ts.npy"
    np.save(path, arr)
    return path


@pytest.fixture
def ts_npy_only_x(tmp_path):
    arr = np.random.rand(8, 30)
    path = tmp_path / "ts_x.npy"
    np.save(path, arr)
    return path


class TestLoadTimeseriesNpy:
    def test_shape_with_target(self, ts_npy_with_target):
        cfg = TimeSeriesDatasetConfig(type="timeseries", path=ts_npy_with_target)
        ds = load_timeseries(cfg)
        assert ds.X.shape == (20, 10)
        assert ds.y.shape == (20,)

    def test_feature_names_are_t0_tN(self, ts_npy_with_target):
        cfg = TimeSeriesDatasetConfig(type="timeseries", path=ts_npy_with_target)
        ds = load_timeseries(cfg)
        assert ds.feature_names == [f"t{i}" for i in range(10)]

    def test_data_type_is_timeseries(self, ts_npy_with_target):
        cfg = TimeSeriesDatasetConfig(type="timeseries", path=ts_npy_with_target)
        ds = load_timeseries(cfg)
        assert ds.data_type == "timeseries"

    def test_only_x(self, ts_npy_only_x):
        cfg = TimeSeriesDatasetConfig(type="timeseries", path=ts_npy_only_x, only_x=True)
        ds = load_timeseries(cfg)
        assert ds.X.shape == (8, 30)
        assert len(ds.y) == 0


# ---------------------------------------------------------------------------
# Routing via load_dataset
# ---------------------------------------------------------------------------

from pulsetrace.data import load_dataset
from sklearn.linear_model import LogisticRegression, LinearRegression
from pulsetrace.adapters.sklearn import SklearnAdapter
from pulsetrace.explainers.lime import LimeExplainer
from pulsetrace.explainers.shap import ShapExplainer
from pulsetrace.explainers.result import ExplanationResult


class TestLoadDatasetRouting:
    def test_routes_timeseries_to_loader(self, ts_npy_only_x):
        cfg = TimeSeriesDatasetConfig(type="timeseries", path=ts_npy_only_x, only_x=True)
        ds = load_dataset(cfg)
        assert isinstance(ds, TimeSeriesDataset)
        assert ds.data_type == "timeseries"
        assert ds.X.shape == (8, 30)


# ---------------------------------------------------------------------------
# TimeSeriesDataset class
# ---------------------------------------------------------------------------

class TestTimeSeriesDataset:
    def test_n_timesteps(self):
        X = np.zeros((10, 30))
        ds = TimeSeriesDataset(
            X=X, y=np.array([], dtype=object),
            feature_names=[f"t{i}" for i in range(30)],
            target_name="", classes=None,
            data_type="timeseries",
        )
        assert ds.n_timesteps == 30

    def test_is_subclass_of_dataset(self):
        X = np.zeros((5, 20))
        ds = TimeSeriesDataset(
            X=X, y=np.array([], dtype=object),
            feature_names=[f"t{i}" for i in range(20)],
            target_name="", classes=None,
            data_type="timeseries",
        )
        assert isinstance(ds, Dataset)

    def test_loader_returns_timeseries_dataset(self, ts_npy_with_target):
        cfg = TimeSeriesDatasetConfig(type="timeseries", path=ts_npy_with_target)
        ds = load_timeseries(cfg)
        assert isinstance(ds, TimeSeriesDataset)
        assert ds.n_timesteps == 10

    def test_data_type_is_timeseries(self):
        X = np.zeros((4, 15))
        ds = TimeSeriesDataset(
            X=X, y=np.array([], dtype=object),
            feature_names=[f"t{i}" for i in range(15)],
            target_name="", classes=None,
            data_type="timeseries",
        )
        assert ds.data_type == "timeseries"


# ---------------------------------------------------------------------------
# Shared fixtures for explainer tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ts_classification_adapter_and_dataset():
    rng = np.random.RandomState(0)
    X = rng.rand(40, 30)  # 40 samples, 30 timesteps
    y = (X[:, 15] > 0.5).astype(int)  # label depends on midpoint
    classes = np.unique(y).astype(object)
    dataset = TimeSeriesDataset(
        X=X, y=y.astype(object),
        feature_names=[f"t{i}" for i in range(30)],
        target_name="label", classes=classes,
        data_type="timeseries",
    )
    model = LogisticRegression(max_iter=200).fit(X, y)
    return SklearnAdapter(model), dataset


@pytest.fixture(scope="module")
def ts_regression_adapter_and_dataset():
    rng = np.random.RandomState(1)
    X = rng.rand(40, 30)
    y = X[:, 0] * 3 + X[:, 29] * 1.5
    dataset = TimeSeriesDataset(
        X=X, y=y.astype(object),
        feature_names=[f"t{i}" for i in range(30)],
        target_name="value", classes=None,
        data_type="timeseries",
    )
    model = LinearRegression().fit(X, y)
    return SklearnAdapter(model), dataset


_LIME_TS_CFG = ExplainerConfig(type="lime", num_features=5, num_samples=100, n_segments=6)
_SHAP_TS_CFG = ExplainerConfig(type="shap", num_features=5)


# ---------------------------------------------------------------------------
# LIME time series tests
# ---------------------------------------------------------------------------

class TestLimeTimeseriesGlobal:
    def test_returns_explanation_result(self, ts_classification_adapter_and_dataset):
        adapter, dataset = ts_classification_adapter_and_dataset
        result = LimeExplainer(_LIME_TS_CFG).explain_global(adapter, dataset)
        assert isinstance(result, ExplanationResult)

    def test_mode_is_global(self, ts_classification_adapter_and_dataset):
        adapter, dataset = ts_classification_adapter_and_dataset
        result = LimeExplainer(_LIME_TS_CFG).explain_global(adapter, dataset)
        assert result.mode == "global"

    def test_contributions_use_segment_names(self, ts_classification_adapter_and_dataset):
        adapter, dataset = ts_classification_adapter_and_dataset
        result = LimeExplainer(_LIME_TS_CFG).explain_global(adapter, dataset)
        assert all(c.feature.startswith("seg_") for c in result.contributions)

    def test_regression_has_base_value(self, ts_regression_adapter_and_dataset):
        adapter, dataset = ts_regression_adapter_and_dataset
        result = LimeExplainer(_LIME_TS_CFG).explain_global(adapter, dataset)
        assert result.base_values is not None
        assert None in result.base_values


class TestLimeTimeseriesLocal:
    def test_local_classification(self, ts_classification_adapter_and_dataset):
        adapter, dataset = ts_classification_adapter_and_dataset
        instance = TimeSeriesDataset(
            X=dataset.X[:1], y=np.array([], dtype=object),
            feature_names=dataset.feature_names,
            target_name="", classes=None,
            data_type="timeseries",
        )
        result = LimeExplainer(_LIME_TS_CFG).explain_local(adapter, instance, dataset)
        assert result.mode == "local"
        assert all(c.feature.startswith("seg_") for c in result.contributions)

    def test_local_regression(self, ts_regression_adapter_and_dataset):
        adapter, dataset = ts_regression_adapter_and_dataset
        instance = TimeSeriesDataset(
            X=dataset.X[:1], y=np.array([], dtype=object),
            feature_names=dataset.feature_names,
            target_name="", classes=None,
            data_type="timeseries",
        )
        result = LimeExplainer(_LIME_TS_CFG).explain_local(adapter, instance, dataset)
        assert result.mode == "local"
        assert result.base_values is not None


# ---------------------------------------------------------------------------
# SHAP time series tests
# ---------------------------------------------------------------------------

class TestShapTimeseriesGlobal:
    def test_returns_explanation_result(self, ts_classification_adapter_and_dataset):
        adapter, dataset = ts_classification_adapter_and_dataset
        result = ShapExplainer(_SHAP_TS_CFG).explain_global(adapter, dataset)
        assert isinstance(result, ExplanationResult)

    def test_mode_is_global(self, ts_classification_adapter_and_dataset):
        adapter, dataset = ts_classification_adapter_and_dataset
        result = ShapExplainer(_SHAP_TS_CFG).explain_global(adapter, dataset)
        assert result.mode == "global"

    def test_contributions_use_timestep_names(self, ts_classification_adapter_and_dataset):
        adapter, dataset = ts_classification_adapter_and_dataset
        result = ShapExplainer(_SHAP_TS_CFG).explain_global(adapter, dataset)
        assert all(c.feature.startswith("t") for c in result.contributions)

    def test_at_most_num_features_contributions(self, ts_classification_adapter_and_dataset):
        adapter, dataset = ts_classification_adapter_and_dataset
        result = ShapExplainer(_SHAP_TS_CFG).explain_global(adapter, dataset)
        assert len(result.contributions) <= _SHAP_TS_CFG.num_features

    def test_regression_has_base_value(self, ts_regression_adapter_and_dataset):
        adapter, dataset = ts_regression_adapter_and_dataset
        result = ShapExplainer(_SHAP_TS_CFG).explain_global(adapter, dataset)
        assert result.base_values is not None


class TestShapTimeseriesLocal:
    def test_local_classification(self, ts_classification_adapter_and_dataset):
        adapter, dataset = ts_classification_adapter_and_dataset
        instance = TimeSeriesDataset(
            X=dataset.X[:1], y=np.array([], dtype=object),
            feature_names=dataset.feature_names,
            target_name="", classes=None,
            data_type="timeseries",
        )
        result = ShapExplainer(_SHAP_TS_CFG).explain_local(adapter, instance, dataset)
        assert result.mode == "local"
        assert len(result.contributions) <= _SHAP_TS_CFG.num_features

    def test_local_regression(self, ts_regression_adapter_and_dataset):
        adapter, dataset = ts_regression_adapter_and_dataset
        instance = TimeSeriesDataset(
            X=dataset.X[:1], y=np.array([], dtype=object),
            feature_names=dataset.feature_names,
            target_name="", classes=None,
            data_type="timeseries",
        )
        result = ShapExplainer(_SHAP_TS_CFG).explain_local(adapter, instance, dataset)
        assert result.mode == "local"
        assert result.base_values is not None


# ---------------------------------------------------------------------------
# End-to-end pipeline smoke test
# ---------------------------------------------------------------------------

import yaml
import joblib
from pulsetrace.main import run


class TestTimeseriesEndToEnd:
    def test_shap_global_regression_pipeline(self, tmp_path, capsys):
        from sklearn.linear_model import LinearRegression

        # Create a synthetic npy dataset (regression: last col is target)
        rng = np.random.RandomState(7)
        X = rng.rand(30, 20)
        y = X[:, 0] * 2 + X[:, 10]
        arr = np.column_stack([X, y])
        data_path = tmp_path / "ts_reg.npy"
        np.save(data_path, arr)

        # Train and save a tiny linear regression model
        model = LinearRegression().fit(X, y)
        model_path = tmp_path / "ts_model.pkl"
        joblib.dump(model, model_path)

        # Write config
        cfg = {
            "model": {"type": "sklearn", "path": str(model_path)},
            "dataset": {"type": "timeseries", "path": str(data_path)},
            "explainer": {"type": "shap", "num_features": 5},
        }
        cfg_path = tmp_path / "ts_cfg.yaml"
        cfg_path.write_text(yaml.dump(cfg))

        capsys.readouterr()
        run(cfg_path)
        out = capsys.readouterr().out

        assert "SHAP" in out
        assert "mode=global" in out
        assert "task=regression" in out
