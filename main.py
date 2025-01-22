import argparse
from pathlib import Path
import typing as ty

import yaml


if ty.TYPE_CHECKING:

    class PulseTraceParser(argparse.Namespace):
        config: Path = Path()


class PulseTracer:
    config: dict[str, ty.Any] = {}

    def init_model(self) -> ty.Any: ...

    def main(self) -> None:
        parser = argparse.ArgumentParser(description="PulseTrace parser")

        _ = parser.add_argument("config", help="Config for model")

        args: PulseTraceParser = ty.cast(PulseTraceParser, parser.parse_args())

        with open(args.config, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        model: ty.Any = self.init_model()


if __name__ == "__main__":
    PulseTracer().main()
