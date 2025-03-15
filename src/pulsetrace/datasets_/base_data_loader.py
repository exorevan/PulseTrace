import typing as ty
from abc import ABC

import numpy as np
import numpy.typing as npt
import pandas as pd

if ty.TYPE_CHECKING:
    from src.pulsetrace.pltypes.config import DatasetPulseTraceConfig


@ty.final
class PTDataSet:
    """Dataset container for machine learning data and targets."""

    __slots__ = (
        "_classes", "_classes_num", "_columns", "_data", "_indexes", "_target", "_target_name"
    )

    def __init__(self, data: pd.DataFrame, target: pd.Series) -> None:
        """
        Initialize dataset from pandas DataFrame and Series.

        Args:
            data: Feature data as pandas DataFrame
            target: Target values as pandas Series

        """
        self.columns = data.columns
        self.data = data
        self.indexes = data.index
        self.target = target
        self.target_name = target.name

        self._classes = np.unique(self.target)
        self._classes_num = len(self._classes)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.target)

    def __getitem__(self, key) -> npt.NDArray[np.float64] | None:
        """Access data by index."""
        if isinstance(key, int):
            return ty.cast(npt.NDArray[np.float64], self.data[key])

        return None

    @property
    def data(self) -> npt.NDArray[np.float64]:
        """Get the feature data array."""
        return self._data

    @data.setter
    def data(self, x: ty.Any) -> None:
        """Set the feature data array, converting to numpy if needed."""
        if isinstance(x, pd.DataFrame):
            self._data = x.to_numpy(dtype=float)
        else:
            self._data = np.asarray(x, dtype=float)

    @property
    def indexes(self) -> list[int | str]:
        """Get the row indexes."""
        return self._indexes

    @indexes.setter
    def indexes(self, x: ty.Any) -> None:
        """Set the row indexes."""
        self._indexes = list(x)

    @property
    def columns(self) -> list[int | str]:
        """Get column names."""
        return self._columns

    @columns.setter
    def columns(self, x: ty.Any) -> None:
        """Set column names."""
        self._columns = list(x)

    @property
    def feature_names(self) -> list[int | str]:
        """Get feature names (alias for columns)."""
        return self.columns

    @property
    def target(self) -> npt.NDArray[np.float64]:
        """Get the target values."""
        return self._target

    @target.setter
    def target(self, x: ty.Any) -> None:
        """Set the target values."""
        if isinstance(x, pd.Series):
            self._target = x.to_numpy(dtype=object)
        else:
            self._target = np.asarray(x, dtype=object)

    @property
    def target_name(self) -> str:
        """Get the target column name."""
        return self._target_name

    @target_name.setter
    def target_name(self, x: ty.Any) -> None:
        """Set the target column name."""
        self._target_name = str(x)

    @property
    def classes(self) -> npt.NDArray[ty.Any]:
        """Get the unique target classes."""
        return self._classes

    @property
    def classes_num(self) -> int:
        """Get the number of unique target classes."""
        return self._classes_num


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""

    config: "DatasetPulseTraceConfig"

    def __init__(self, config: "DatasetPulseTraceConfig") -> None:
        """
        Initialize with configuration.

        Args:
            config: Dataset configuration parameters

        """
        self.config = config
