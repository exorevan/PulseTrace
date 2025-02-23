import os
import typing as ty

import pandas as pd

from datasets.base_data_loader import BaseDataLoader, PTDataSet

if ty.TYPE_CHECKING:
    from pltypes.config import DatasetPulseTraceConfig


class CSVDataLoader(BaseDataLoader):
    def __init__(self, config: "DatasetPulseTraceConfig"):
        super().__init__(config)

    def load_data(self, path: str | None = None, input: bool = False) -> PTDataSet:
        file_path = path if path else self.config.get("path")

        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        csv_params = self.config.get("csv_params", {})
        delimiter = csv_params.get("delimiter", ",")
        index_col = csv_params.get("index_col", None)
        header = csv_params.get("header", None)

        data = pd.read_csv(
            file_path, delimiter=delimiter, index_col=index_col, header=header
        )

        if input:
            return PTDataSet(data, pd.Series())
        else:
            return PTDataSet(
                ty.cast(pd.DataFrame, data.iloc[:, :-1]),
                ty.cast(pd.Series, data.iloc[:, -1]),
            )
