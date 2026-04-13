from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pulsetrace.config.schema import CsvDatasetConfig

from .dataset import Dataset


def load_csv(config: CsvDatasetConfig) -> Dataset:
    """Load a CSV file and return a Dataset.

    When config.only_x is True, the entire file is treated as features
    (no target column). Use this for local-explanation input instances.
    """
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(
        path,
        delimiter=config.delimiter,
        header=config.header,
        index_col=config.index_col,
    )
    # Ensure all column names are strings
    df.columns = [str(c) for c in df.columns]

    if config.only_x:
        return Dataset(
            X=df.to_numpy(dtype=float),
            y=np.array([], dtype=object),
            feature_names=list(df.columns),
            target_name="",
            classes=None,
        )

    X_df = df.iloc[:, :-1]
    y_series = df.iloc[:, -1]

    return Dataset(
        X=X_df.to_numpy(dtype=float),
        y=y_series.to_numpy(dtype=object),
        feature_names=list(X_df.columns),
        target_name=str(y_series.name) if y_series.name is not None else "target",
        classes=np.unique(y_series.to_numpy(dtype=object)),
    )
