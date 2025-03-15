import argparse
import logging
import sys
import typing as ty
from pathlib import Path

import yaml

from datasets_ import CSVDataLoader, ImageDataLoader, TextDataLoader
from pulsetrace.explainers import LimeExplainer, ShapExplainer
from model_loaders import PyTorchModelLoader, SklearnModelLoader, TensorFlowModelLoader
from utils.logger import setup_logging

if ty.TYPE_CHECKING:
    from datasets.base_data_loader import PTDataSet
    from explainers.base_explainer import BaseExplainer
    from pltypes import PLDataLoader, PLExplainer, PLModelLoader
    from pltypes.config import (
        DatasetPulseTraceConfig,
        ExplainerPulseTraceConfig,
        ModelPulseTraceConfig,
        PulseTraceConfig,
    )
    from pltypes.models import PLModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explainable Machine Learning Model Interpreter Application"
    )
    _ = parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to the configuration YAML file (e.g., config/config.yaml)",
    )

    return parser.parse_args()


def load_configuration(cfg_path: Path) -> "PulseTraceConfig":
    try:
        with open(cfg_path, "r") as stream:
            config = ty.cast("PulseTraceConfig", yaml.safe_load(stream))

        return config

    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)


def get_model_loader(model_config: "ModelPulseTraceConfig") -> "PLModelLoader":
    model_type = model_config.get("type", "").lower()

    if model_type in {"tf", "keras"}:
        return TensorFlowModelLoader(model_config)
    elif model_type == "pt":
        return PyTorchModelLoader(model_config)
    elif model_type == "sklearn":
        return SklearnModelLoader(model_config)
    else:
        logging.error("Unsupported model type specified. Use 'tf', 'pt', or 'sklearn'.")
        raise TypeError(
            "Unsupported model type specified. Use 'tf', 'pt', or 'sklearn'."
        )


def get_dataset_loader(dataset_config: "DatasetPulseTraceConfig") -> "PLDataLoader":
    dataset_type = dataset_config.get("type", "").lower()

    if dataset_type == "csv":
        return CSVDataLoader(dataset_config)
    elif dataset_type == "image":
        return ImageDataLoader(dataset_config)
    elif dataset_type == "text":
        return TextDataLoader(dataset_config)
    else:
        logging.error(
            "Unsupported dataset type specified. Use 'csv', 'image', or 'text'."
        )
        sys.exit(1)


def get_explainer(explainer_config: "ExplainerPulseTraceConfig") -> "PLExplainer":
    explainer_type = explainer_config.get("type", "").lower()

    if explainer_type == "lime":
        return LimeExplainer(explainer_config)
    elif explainer_type == "shap":
        return ShapExplainer(explainer_config)
    else:
        logging.error("Unsupported explainer type specified. Use 'lime' or 'shap'.")
        sys.exit(1)


def run_global_explanation(
    model: "PLModel", explainer: "BaseExplainer", dataset: "PTDataSet"
) -> dict[ty.Literal["global_explanation"], ty.Any]:
    logging.info("Running global explanation...")

    result = explainer.explain_global(model, dataset)
    logging.info("Global explanation completed.")

    return result


def run_local_explanation(
    model: "PLModel",
    explainer: "BaseExplainer",
    input_instance: "PTDataSet",
    dataset: "PTDataSet",
) -> dict[ty.Literal["local_explanation"], ty.Any]:
    logging.info("Running local explanation for the provided input instance...")

    result = explainer.explain_local(model, input_instance, dataset)
    logging.info("Local explanation completed.")

    return result


def main() -> None:
    args = parse_args()
    config = load_configuration(args.cfg)

    logging_config = config.get("logging", {})
    setup_logging(logging_config)

    logging.info("Configuration loaded successfully.")

    model_config = config.get("model", {})
    model_loader = get_model_loader(model_config)
    model = model_loader.load_model()
    logging.info("Model loaded successfully.")

    dataset_config = config.get("dataset", {})
    dataset_loader = get_dataset_loader(dataset_config)
    dataset = dataset_loader.load_data()

    logging.info("Dataset loaded successfully.")

    explainer_config = config.get("explainer", {})
    explainer = get_explainer(explainer_config)

    app_config = config.get("app", {})
    mode = app_config.get("mode", "global").lower()

    if mode == "global":
        results = run_global_explanation(model, explainer, dataset)
    elif mode == "local":
        local_config = config.get("local", {})

        input_instance = get_dataset_loader(local_config.get("dataset", {})).load_data(
            input=True
        )
        results = run_local_explanation(model, explainer, input_instance, dataset)
    else:
        logging.error(
            "Invalid application mode specified in configuration. Use 'global' or 'local'."
        )
        sys.exit(1)

    logging.info("Explanation Results:")
    logging.info(results)


if __name__ == "__main__":
    main()
