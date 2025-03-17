import os
from pathlib import Path
import typing as ty

from pulsetrace.datasets_.base_data_loader import BaseDataLoader

if ty.TYPE_CHECKING:
    from pulsetrace.pltypes.config import DatasetPulseTraceConfig


class TextDataLoader(BaseDataLoader):
    def __init__(self, config: "DatasetPulseTraceConfig"):
        super().__init__(config)

    def load_data(self, path: str | None = None):
        file_path = path if path else self.config.get("path")

        if not file_path or not Path(file_path).exists():
            msg = f"Text file not found: {file_path}"
            raise FileNotFoundError(msg)

        with Path(file_path).open("r") as f:
            data = f.readlines()

        return [line.strip() for line in data]
