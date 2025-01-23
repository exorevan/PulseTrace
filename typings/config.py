import typing as ty

if ty.TYPE_CHECKING:

    class PulseTraceConfigExplanation(ty.TypedDict):
        method: str

    class PulseTraceConfigInput(ty.TypedDict):
        class_names: list[str]
        input_type: str
        feature_names: list[str]
        function: str
        values: list[ty.Any]

    class PulseTraceConfigModel(ty.TypedDict):
        model_obj: str
        module: str
        weights: str

    class PulseTraceConfigOutput(ty.TypedDict):
        name: str
        path: str

    class PulseTraceConfig(ty.TypedDict):
        explanation: PulseTraceConfigExplanation
        input: PulseTraceConfigInput
        model: PulseTraceConfigModel
        output: PulseTraceConfigOutput
