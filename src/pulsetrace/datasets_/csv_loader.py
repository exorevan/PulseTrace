import typing as ty
from pathlib import Path

import pandas as pd

from .base_data_loader import BaseDataLoader, PTDataSet

if ty.TYPE_CHECKING:
    from src.pulsetrace.datasets_.base_data_loader import PTDataSet
    from src.pulsetrace.pltypes.config import DatasetPulseTraceConfig


class CSVDataLoader(BaseDataLoader):
    def __init__(self, config: "DatasetPulseTraceConfig") -> None:
        super().__init__(config)

    def load_data(self, path: str | None = None, *, input_: bool = False) -> PTDataSet:
        file_path = path if path else self.config.get("path")

        if not file_path or not Path(file_path).exists():
            msg = f"CSV file not found: {file_path}"
            raise FileNotFoundError(msg)

        csv_params = self.config.get("csv_params", {})
        delimiter = csv_params.get("delimiter", ",")
        index_col = csv_params.get("index_col", None)
        header = csv_params.get("header", None)
        only_x = csv_params.get("only_x", False)

        data = pd.read_csv(
            file_path, delimiter=delimiter, index_col=index_col, header=header
        )

        if input_ or only_x:
            return PTDataSet(data, pd.Series())

        return PTDataSet(
            data.iloc[:, :-1],
            data.iloc[:, -1],
        )
