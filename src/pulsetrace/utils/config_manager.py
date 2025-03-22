import typing as ty
from pathlib import Path

import yaml


class ConfigManager:
    config_path: Path
    config: dict[str, ty.Any]

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict[str, ty.Any]:
        try:
            with open(self.config_path) as stream:
                config = yaml.safe_load(stream)

                if not isinstance(config, dict):
                    raise ValueError("Configuration must be a dictionary.")

                return ty.cast(dict[str, ty.Any], config)

        except Exception as e:
            raise Exception(
                f"Failed to load configuration from {self.config_path}: {e}"
            ) from e

    def get(self, key: str, default: ty.Any = None) -> ty.Any:
        return self.config.get(key, default)
