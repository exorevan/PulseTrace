from pulsetrace.config.schema import CsvDatasetConfig

from .csv_loader import load_csv
from .dataset import Dataset


def load_dataset(config: CsvDatasetConfig) -> Dataset:
    return load_csv(config)


__all__ = ["Dataset", "load_dataset", "load_csv"]