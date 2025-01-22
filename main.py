import argparse
import typing as ty


def init_model() -> ty.Any: ...


def main() -> None:
    parser = argparse.ArgumentParser(description="PulseTrace parser")

    _ = parser.add_argument("config", help="Config for model")

    args = parser.parse_args()

    model: ty.Any = init_model(args.config)


if __name__ == "__main__":
    main()
