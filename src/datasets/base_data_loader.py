import typing as ty
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd

if ty.TYPE_CHECKING:
    from types.config import DatasetPulseTraceConfig


class PTDataSet:
    _columns: list[int | str]
    _data: npt.NDArray[np.float64]
    _indexes: list[int | str]
    _target: npt.NDArray[np.float64]
    _target_name: str
    _feature_names: list[str] = []

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.columns = data.columns
        self.data = data.values
        self.indexes = data.index
        self.target = target
        self.target_name = target.name

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, key):
        if isinstance(key, int):
            return ty.cast(npt.NDArray[np.float64], self.data[key])

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return self._data

    @data.setter
    def data(self, x: ty.Any) -> None:
        self._data = np.array(x, dtype=float)

    @property
    def indexes(self) -> list[int | str]:
        return self._indexes

    @indexes.setter
    def indexes(self, x: ty.Any) -> None:
        self._indexes = list(x)

    @property
    def columns(self) -> list[int | str]:
        return self._columns

    @columns.setter
    def columns(self, x: ty.Any) -> None:
        self._columns = list(x)

    @property
    def feature_names(self) -> list[int | str]:
        return self.columns

    @property
    def target(self) -> npt.NDArray[np.float64]:
        return self._target

    @target.setter
    def target(self, x: ty.Any) -> None:
        self._target = np.array(x, dtype=object)

    @property
    def target_name(self) -> str:
        return self._target_name

    @target_name.setter
    def target_name(self, x: ty.Any) -> None:
        self._target_name = str(x)

    def get_x(self) -> npt.NDArray[np.float64]:
        return self.data

    def get_y(self) -> npt.NDArray[np.float64]:
        return self.target


class BaseDataLoader(ABC):
    config: "DatasetPulseTraceConfig"

    def __init__(self, config: "DatasetPulseTraceConfig"):
        self.config = config
