from __future__ import annotations

import typing as ty
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SklearnModelConfig(_Base):
    type: ty.Literal["sklearn"]
    path: Path


class KerasModelConfig(_Base):
    type: ty.Literal["keras", "tf"]
    path: Path


class TorchArchConfig(_Base):
    path: Path
    class_name: str
    init_params: dict[str, ty.Any] = Field(default_factory=dict)


class TorchModelConfig(_Base):
    type: ty.Literal["pt"]
    weights_path: Path
    architecture: TorchArchConfig
    task: ty.Literal["classification", "regression"]


ModelConfig = ty.Annotated[
    SklearnModelConfig | KerasModelConfig | TorchModelConfig,
    Field(discriminator="type"),
]


class CsvDatasetConfig(_Base):
    type: ty.Literal["csv"]
    path: Path
    delimiter: str = ","
    header: int | None = 0
    index_col: str | int | None = None
    only_x: bool = False


# Single type for now; becomes a discriminated union when image/text loaders are added
DatasetConfig = CsvDatasetConfig


class ExplainerConfig(_Base):
    type: ty.Literal["lime", "shap"]
    num_features: int = 10
    num_samples: int = 5000  # used by LIME only


class LocalConfig(_Base):
    dataset: DatasetConfig


class AppConfig(_Base):
    mode: ty.Literal["global", "local"] = "global"
    output_format: ty.Literal["console"] = "console"


class LoggingConfig(_Base):
    level: ty.Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Path | None = None


class PulseTraceConfig(_Base):
    app: AppConfig = Field(default_factory=AppConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    model: ModelConfig
    dataset: DatasetConfig
    explainer: ExplainerConfig
    local: LocalConfig | None = None

    @model_validator(mode="after")
    def local_required_in_local_mode(self) -> "PulseTraceConfig":
        if self.app.mode == "local" and self.local is None:
            raise ValueError("'local' section is required when app.mode is 'local'")
        return self
