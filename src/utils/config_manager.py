from pathlib import Path

import yaml


class ConfigManager:
    config_path: Path

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, "r") as stream:
                return yaml.safe_load(stream)
        except Exception as e:
            raise Exception(
                f"Failed to load configuration from {self.config_path}: {e}"
            )

    def get(self, key, default=None):
        return self.config.get(key, default)
