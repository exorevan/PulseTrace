from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pulsetrace.config.schema import TimeSeriesDatasetConfig

from .dataset import Dataset
from .timeseries_dataset import TimeSeriesDataset


def load_timeseries(config: TimeSeriesDatasetConfig) -> TimeSeriesDataset:
    """Load a time series dataset from a CSV or NumPy .npy file.

    CSV: rows = samples, columns = timesteps (plus optional target column).
    NumPy: shape (n_samples, n_timesteps) with no target, or (n_samples, n_timesteps+1)
           where the last column is the target when only_x=False.
    """
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"Time series file not found: {path}")

    if path.suffix == ".npy":
        return _load_npy(config, path)
    return _load_csv(config, path)


def _load_csv(config: TimeSeriesDatasetConfig, path: Path) -> TimeSeriesDataset:
    df = pd.read_csv(path)

    if config.only_x:
        X_df = df
        y = np.array([], dtype=object)
        target_name = ""
        classes = None
    elif config.target_col is not None:
        y = df[config.target_col].to_numpy(dtype=object)
        X_df = df.drop(columns=[config.target_col])
        target_name = config.target_col
        classes = np.unique(y)
    else:
        # No target_col specified and not only_x: treat last column as target
        y = df.iloc[:, -1].to_numpy(dtype=object)
        X_df = df.iloc[:, :-1]
        target_name = str(df.columns[-1])
        classes = np.unique(y)

    n_timesteps = X_df.shape[1]
    if config.n_timesteps is not None and n_timesteps != config.n_timesteps:
        raise ValueError(
            f"Expected {config.n_timesteps} timesteps, got {n_timesteps}"
        )

    feature_names = [f"t{i}" for i in range(n_timesteps)]
    X = X_df.to_numpy(dtype=float)

    return TimeSeriesDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        target_name=target_name,
        classes=classes,
        data_type="timeseries",
    )


def _load_npy(config: TimeSeriesDatasetConfig, path: Path) -> TimeSeriesDataset:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D array (n_samples, n_timesteps[+1]), got shape {arr.shape}"
        )

    if config.only_x:
        X = arr.astype(float)
        y = np.array([], dtype=object)
        target_name = ""
        classes = None
    else:
        X = arr[:, :-1].astype(float)
        y = arr[:, -1].astype(object)
        target_name = "target"
        classes = np.unique(y)

    n_timesteps = X.shape[1]
    if config.n_timesteps is not None and n_timesteps != config.n_timesteps:
        raise ValueError(
            f"Expected {config.n_timesteps} timesteps, got {n_timesteps}"
        )

    feature_names = [f"t{i}" for i in range(n_timesteps)]

    return TimeSeriesDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        target_name=target_name,
        classes=classes,
        data_type="timeseries",
    )
