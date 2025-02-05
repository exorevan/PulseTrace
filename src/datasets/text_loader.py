import os


class TextDataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self, path=None):
        file_path = path if path else self.config.get("path")
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.readlines()
        data = [line.strip() for line in data]
        return data
