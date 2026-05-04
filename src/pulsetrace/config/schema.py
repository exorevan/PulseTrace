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


class HfModelConfig(_Base):
    """HuggingFace AutoModelForSequenceClassification — local path or Hub ID."""

    type: ty.Literal["hf"]
    path_or_name: str
    labels: list[str] = Field(min_length=1)
    max_length: int = 512


ModelConfig = ty.Annotated[
    SklearnModelConfig | KerasModelConfig | TorchModelConfig | HfModelConfig,
    Field(discriminator="type"),
]


class CsvDatasetConfig(_Base):
    type: ty.Literal["csv"]
    path: Path
    delimiter: str = ","
    header: int | None = 0
    index_col: str | int | None = None
    only_x: bool = False


class TimeSeriesDatasetConfig(_Base):
    type: ty.Literal["timeseries"]
    path: Path
    n_timesteps: int | None = None   # validated against actual shape when provided
    target_col: str | None = None    # CSV only: name of the label column
    only_x: bool = False             # true = no target column (local mode input)


class ImageDatasetConfig(_Base):
    type: ty.Literal["image"]
    path: Path
    image_size: list[int] | None = None   # [H, W]; validated as 2-element list
    only_x: bool = False

    @model_validator(mode="after")
    def _validate_image_size(self) -> "ImageDatasetConfig":
        if self.image_size is not None and len(self.image_size) != 2:
            raise ValueError("image_size must have exactly 2 elements [H, W]")
        return self


class BuiltinDatasetConfig(_Base):
    type: ty.Literal["builtin"]
    name: str
    only_x: bool = False
    split: ty.Literal["train", "test"] = "train"   # keras only; ignored for sklearn
    max_samples: int | None = None                  # cap number of samples loaded


class TextDatasetConfig(_Base):
    """Dataset of .txt files organised in one subdirectory per class."""

    type: ty.Literal["text"]
    path: Path
    only_x: bool = False


DatasetConfig = ty.Annotated[
    CsvDatasetConfig | TimeSeriesDatasetConfig | ImageDatasetConfig | BuiltinDatasetConfig | TextDatasetConfig,
    Field(discriminator="type"),
]


class ExplainerConfig(_Base):
    type: ty.Literal["lime", "shap", "ig"]
    num_features: int = 10
    num_samples: int = 5000
    global_samples: int = 10
    n_segments: int = 10
    ig_steps: int = 50
    ig_baseline: ty.Literal["zero", "mean"] = "zero"


class LocalConfig(_Base):
    dataset: DatasetConfig


class AppConfig(_Base):
    mode: ty.Literal["global", "local"] = "global"
    output_format: ty.Literal["console", "json", "html"] = "console"


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
