import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from pulsetrace.config.schema import CsvDatasetConfig
from pulsetrace.data.csv_loader import load_csv
from pulsetrace.data.dataset import Dataset


@pytest.fixture
def csv_with_target(tmp_path):
    df = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 6.2],
        "sepal_width": [3.5, 3.0, 2.9],
        "species": ["setosa", "setosa", "virginica"],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def csv_without_target(tmp_path):
    df = pd.DataFrame({
        "f1": [1.0, 2.0],
        "f2": [3.0, 4.0],
    })
    path = tmp_path / "input.csv"
    df.to_csv(path, index=False)
    return path


def test_load_csv_shape(csv_with_target):
    config = CsvDatasetConfig(type="csv", path=csv_with_target)
    ds = load_csv(config)
    assert ds.X.shape == (3, 2)
    assert ds.y.shape == (3,)


def test_load_csv_feature_names(csv_with_target):
    config = CsvDatasetConfig(type="csv", path=csv_with_target)
    ds = load_csv(config)
    assert ds.feature_names == ["sepal_length", "sepal_width"]


def test_load_csv_target_name(csv_with_target):
    config = CsvDatasetConfig(type="csv", path=csv_with_target)
    ds = load_csv(config)
    assert ds.target_name == "species"


def test_load_csv_classes(csv_with_target):
    config = CsvDatasetConfig(type="csv", path=csv_with_target)
    ds = load_csv(config)
    assert ds.classes is not None
    assert set(ds.classes) == {"setosa", "virginica"}


def test_load_csv_only_x(csv_without_target):
    config = CsvDatasetConfig(type="csv", path=csv_without_target, only_x=True)
    ds = load_csv(config)
    assert ds.X.shape == (2, 2)
    assert len(ds.y) == 0
    assert ds.classes is None
    assert ds.target_name == ""


def test_load_csv_file_not_found():
    config = CsvDatasetConfig(type="csv", path=Path("nonexistent.csv"))
    with pytest.raises(FileNotFoundError):
        load_csv(config)


def test_dataset_len(csv_with_target):
    config = CsvDatasetConfig(type="csv", path=csv_with_target)
    ds = load_csv(config)
    assert len(ds) == 3


def test_dataset_instance(csv_with_target):
    config = CsvDatasetConfig(type="csv", path=csv_with_target)
    ds = load_csv(config)
    row = ds.instance(0)
    assert row.shape == (2,)
    np.testing.assert_array_almost_equal(row, [5.1, 3.5])


def test_dataset_is_frozen(csv_with_target):
    config = CsvDatasetConfig(type="csv", path=csv_with_target)
    ds = load_csv(config)
    with pytest.raises((AttributeError, TypeError)):
        ds.target_name = "changed"  # type: ignore


def test_load_csv_headerless(tmp_path):
    """When header=None, pandas uses integer column indices.
    After stringification, y_series.name is e.g. "2" (truthy) — not None.
    The is-not-None guard prevents falsiness bugs if name were integer 0."""
    df = pd.DataFrame([[1.0, 2.0, "a"], [3.0, 4.0, "b"]])
    path = tmp_path / "no_header.csv"
    df.to_csv(path, index=False, header=False)
    config = CsvDatasetConfig(type="csv", path=path, header=None)
    ds = load_csv(config)
    # Column names are stringified indices; last column becomes "2"
    assert ds.target_name == "2"
    assert ds.X.shape == (2, 2)


def test_load_csv_only_x_y_dtype(csv_without_target):
    """only_x path must return y with dtype=object to match Dataset.y annotation."""
    config = CsvDatasetConfig(type="csv", path=csv_without_target, only_x=True)
    ds = load_csv(config)
    assert ds.y.dtype == object


def test_dataset_default_data_type():
    ds = Dataset(
        X=np.zeros((3, 2)),
        y=np.array(["a", "b", "c"], dtype=object),
        feature_names=["f1", "f2"],
        target_name="label",
        classes=np.array(["a", "b", "c"], dtype=object),
    )
    assert ds.data_type == "tabular"


def test_dataset_custom_data_type():
    ds = Dataset(
        X=np.zeros((3, 10)),
        y=np.array([0, 1, 0], dtype=object),
        feature_names=[f"t{i}" for i in range(10)],
        target_name="label",
        classes=None,
        data_type="timeseries",
    )
    assert ds.data_type == "timeseries"
