import argparse
import traceback
import typing as ty
from pathlib import Path
from pltypes import ModuleType

import sklearn
import tensorflow
import yaml

from pulsetrace.consts import (
    MODEL_EXPLANATION_ACCORDANCE,
    MODEL_LOADER_ACCORDANCE,
    MODEL_MODULE,
)

if ty.TYPE_CHECKING:
    from pulsetrace.model_loader import SUPPORTED_MODELS
    from pulsetrace.typings.config import PulseTraceConfig

    class PulseTraceParser(argparse.Namespace):
        config: Path = Path()


class PulseTracer:
    config: "PulseTraceConfig"
    explanation: bool = False
    model: "SUPPORTED_MODELS | None" = None
    module: ModuleType | None = None

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="PulseTrace parser")

        _ = parser.add_argument("config", help="Config for model")

        args: "PulseTraceParser" = ty.cast("PulseTraceParser", parser.parse_args())

        with open(args.config, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def init_model(self) -> "SUPPORTED_MODELS | None":
        self.module = MODEL_MODULE[self.config["model"]["module"]]

        if self.module is sklearn:
            return MODEL_LOADER_ACCORDANCE[sklearn](
                Path(self.config["model"]["weights"])
            )

        return None

        # TODO:
        # temp_module: ModuleType = MODEL_MODULE[self.config.model.module]

        # for part in self.config.model.model_obj.split("."):
        #     temp_module = getattr(temp_module, part)

        # model_class: type = ty.cast(type, ty.cast(object, temp_module))

    def explain_model(self) -> bool:
        try:
            if self.module is None or self.model is None:
                return False

            MODEL_EXPLANATION_ACCORDANCE[self.config["explanation"]["method"]][
                self.module
            ](self.model, self.config)

            return True

        except Exception as e:
            print(traceback.format_exc())

            return False

    def main(self) -> None:
        self.model = self.init_model()

        if self.model is None:
            raise ValueError("Model not supported")

        self.explanation = self.explain_model()

        if not self.explanation:
            raise RuntimeError("Error while explaining")


if __name__ == "__main__":
    PulseTracer().main()
