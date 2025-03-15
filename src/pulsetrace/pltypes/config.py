import typing as ty
from typing import TypedDict

if ty.TYPE_CHECKING:

    class AppPulseTraceConfig(TypedDict):
        mode: str
        interactive: bool
        output_format: str

    class LoggingPulseTraceConfig(TypedDict):
        level: str
        file: str

    class ModelParametersPulseTraceConfig(TypedDict):
        n_estimators: int
        max_depth: int

    class ModelPulseTraceConfig(TypedDict):
        type: str
        path: str
        parameters: ModelParametersPulseTraceConfig
        random_state: int

    class DatasetCSVParamsPulseTraceConfig(TypedDict):
        delimiter: str
        header: int
        index_col: str
        only_x: bool

    class DatasetPreprocessPulseTraceConfig(TypedDict):
        normalize: bool
        additional_steps: list[ty.Any]

    class DatasetPulseTraceConfig(TypedDict):
        path: str
        type: str
        csv_params: DatasetCSVParamsPulseTraceConfig
        preprocess: DatasetPreprocessPulseTraceConfig

    class ExplainerParametersPulseTraceConfig(TypedDict):
        num_features: int
        num_samples: int
        additional_params: dict[str, ty.Any]

    class ExplainerPulseTraceConfig(TypedDict):
        type: str
        parameters: ExplainerParametersPulseTraceConfig

    class LocalPulseTraceConfig(TypedDict):
        input_path: str

    class PulseTraceConfig(TypedDict):
        app: AppPulseTraceConfig
        logging: LoggingPulseTraceConfig
        model: ModelPulseTraceConfig
        dataset: DatasetPulseTraceConfig
        explainer: ExplainerPulseTraceConfig
        local: LocalPulseTraceConfig
