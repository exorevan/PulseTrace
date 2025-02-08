import os
import typing as ty

from datasets.base_data_loader import BaseDataLoader

if ty.TYPE_CHECKING:
    from types.config import DatasetPulseTraceConfig


class TextDataLoader(BaseDataLoader):
    def __init__(self, config: "DatasetPulseTraceConfig"):
        super().__init__(config)

    def load_data(self, path: str | None = None):
        file_path = path if path else self.config.get("path")

        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = f.readlines()

        data = [line.strip() for line in data]

        return data
