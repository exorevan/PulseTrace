from __future__ import annotations

from pulsetrace.config.schema import (
    BuiltinDatasetConfig,
    CsvDatasetConfig,
    ImageDatasetConfig,
    TextDatasetConfig,
    TimeSeriesDatasetConfig,
)

from .builtin_loader import load_builtin
from .csv_loader import load_csv
from .dataset import Dataset
from .image_dataset import ImageDataset
from .image_loader import load_image_dataset
from .text_dataset import TextDataset
from .text_loader import load_text_dataset
from .timeseries_dataset import TimeSeriesDataset
from .timeseries_loader import load_timeseries


def load_dataset(
    config: CsvDatasetConfig | TimeSeriesDatasetConfig | ImageDatasetConfig | BuiltinDatasetConfig | TextDatasetConfig,
) -> Dataset:
    """Route dataset config to the appropriate loader."""
    if isinstance(config, BuiltinDatasetConfig):
        return load_builtin(config)
    if isinstance(config, ImageDatasetConfig):
        return load_image_dataset(config)
    if isinstance(config, TimeSeriesDatasetConfig):
        return load_timeseries(config)
    if isinstance(config, TextDatasetConfig):
        return load_text_dataset(config)
    return load_csv(config)


__all__ = [
    "Dataset",
    "ImageDataset",
    "TextDataset",
    "TimeSeriesDataset",
    "load_dataset",
    "load_builtin",
    "load_csv",
    "load_image_dataset",
    "load_text_dataset",
    "load_timeseries",
]
