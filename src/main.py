import argparse
import logging
import os
import sys
import typing as ty
from pathlib import Path

import yaml

# Dataset loaders
from datasets.csv_loader import CSVDataLoader
from datasets.image_loader import ImageDataLoader
from datasets.text_loader import TextDataLoader

# Explainer implementations
from explainers.lime_explainer import LimeExplainer
from explainers.shap_explainer import ShapExplainer
from models.pytorch_loader import PyTorchModelLoader
from models.sklearn_loader import SklearnModelLoader

# Model loaders
from models.tensorflow_loader import TensorFlowModelLoader

# Import configuration and logging utilities.
from utils.config_manager import (
    ConfigManager,
)  # Optional: if you want to use a custom config manager.
from utils.logger import setup_logging

if ty.TYPE_CHECKING:
    from pltypes.config import (
        DatasetPulseTraceConfig,
        ModelPulseTraceConfig,
        PulseTraceConfig,
    )


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
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def get_model_loader(model_config: "ModelPulseTraceConfig"):
    model_type = model_config.get("type", "").lower()

    if model_type == "tf":
        return TensorFlowModelLoader(model_config)
    elif model_type == "pt":
        return PyTorchModelLoader(model_config)
    elif model_type == "sklearn":
        return SklearnModelLoader(model_config)
    else:
        print("Unsupported model type specified. Use 'tf', 'pt', or 'sklearn'.")
        sys.exit(1)


def get_dataset_loader(dataset_config: "DatasetPulseTraceConfig"):
    dataset_type = dataset_config.get("type", "").lower()

    if dataset_type == "csv":
        return CSVDataLoader(dataset_config)
    elif dataset_type == "image":
        return ImageDataLoader(dataset_config)
    elif dataset_type == "text":
        return TextDataLoader(dataset_config)
    else:
        print("Unsupported dataset type specified. Use 'csv', 'image', or 'text'.")
        sys.exit(1)


def get_explainer(explainer_config):
    explainer_type = explainer_config.get("type", "").lower()
    if explainer_type == "lime":
        return LimeExplainer(explainer_config)
    elif explainer_type == "shap":
        return ShapExplainer(explainer_config)
    else:
        print("Unsupported explainer type specified. Use 'lime' or 'shap'.")
        sys.exit(1)


def run_global_explanation(model, dataset, explainer):
    logging.info("Running global explanation...")

    # The explain_global method should accept the complete dataset as input.
    result = explainer.explain_global(model, dataset)
    logging.info("Global explanation completed.")

    return result


def run_local_explanation(model, explainer, input_instance, dataset):
    logging.info("Running local explanation for the provided input instance...")

    # The explain_local method should accept a single input instance.
    result = explainer.explain_local(model, input_instance, dataset)
    logging.info("Local explanation completed.")

    return result


def main():
    args = parse_args()
    config = load_configuration(args.cfg)

    # Setup logging based on configuration settings.
    logging_config = config.get("logging", {})
    setup_logging(logging_config)

    logging.info("Configuration loaded successfully.")

    # Load model using its dedicated loader.
    model_config = config.get("model", {})
    model_loader = get_model_loader(model_config)
    model = model_loader.load_model()
    logging.info("Model loaded successfully.")

    # Load dataset using its dedicated loader.
    dataset_config = config.get("dataset", {})
    dataset_loader = get_dataset_loader(dataset_config)
    dataset = dataset_loader.load_data()
    logging.info("Dataset loaded successfully.")

    # Initialize the explainer.
    explainer_config = config.get("explainer", {})
    explainer = get_explainer(explainer_config)

    # Decide which explanation mode to run from the 'app' section.
    app_config = config.get("app", {})
    mode = app_config.get("mode", "global").lower()

    if mode == "global":
        results = run_global_explanation(model, dataset, explainer)
    elif mode == "local":
        local_config = config.get("local", {})

        # For local explanation, assume that the same dataset loader can be reused to load a single input
        # instance from the provided path (or you could implement a specialized loader method).
        input_instance = get_dataset_loader(local_config.get("dataset", {})).load_data(
            input=True
        )
        results = run_local_explanation(model, explainer, input_instance, dataset)
    else:
        logging.error(
            "Invalid application mode specified in configuration. Use 'global' or 'local'."
        )
        sys.exit(1)

    # Output the explanation results.
    print("Explanation Results:")
    print(results)


if __name__ == "__main__":
    main()
