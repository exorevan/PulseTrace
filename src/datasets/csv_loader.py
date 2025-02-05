import os

import pandas as pd


class CSVDataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self, path=None):
        file_path = path if path else self.config.get("path")
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        csv_params = self.config.get("csv_params", {})
        delimiter = csv_params.get("delimiter", ",")
        index_col = csv_params.get("index_col", None)
        header = csv_params.get("header", 0)
        data = pd.read_csv(
            file_path, delimiter=delimiter, index_col=index_col, header=header
        )
        return data
