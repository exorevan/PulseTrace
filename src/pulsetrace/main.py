import argparse
import sys
import typing as ty
from pathlib import Path

import yaml
from datasets_ import CSVDataLoader, ImageDataLoader, TextDataLoader
from explainers import LimeExplainer, ShapExplainer
from logger import pllogger
from model_loaders import PyTorchModelLoader, SklearnModelLoader, TensorFlowModelLoader
from utils.logger import setup_logging

if ty.TYPE_CHECKING:
    from src.pulsetrace.datasets_.base_data_loader import PTDataSet
    from src.pulsetrace.explainers.base_explainer import BaseExplainer
    from src.pulsetrace.pltypes import PLExplainer, PLModelLoader
    from src.pulsetrace.pltypes.config import (
        DatasetPulseTraceConfig,
        ExplainerPulseTraceConfig,
        ModelPulseTraceConfig,
        PulseTraceConfig,
    )
    from src.pulsetrace.pltypes.models import PLModel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
    """Load and parse the configuration YAML file."""
    try:
        with Path.open(cfg_path, "r") as stream:
            return ty.cast("PulseTraceConfig", yaml.safe_load(stream))
    except FileNotFoundError:
        pllogger.exception(f"Configuration file not found: {cfg_path}")
        raise
    except yaml.YAMLError as e:
        pllogger.exception(f"Invalid YAML in configuration: {e}")
        raise
    except Exception as e:
        pllogger.exception(f"Unexpected error loading configuration: {e}")
        raise


def create_component(config: dict[str, ty.Any], component_type: str, components_map: dict[str, ty.Any]) -> ty.Any:
    """Create components based on configuration."""
    component_id = config.get("type", "").lower()
    component_class = components_map.get(component_id)

    if not component_class:
        valid_types = ", ".join(f"'{t}'" for t in components_map)
        msg = f"Unsupported {component_type} type: '{component_id}'. Valid types: {valid_types}."
        pllogger.error(msg)
        raise ValueError(msg)

    return component_class(config)


def get_model_loader(model_config: "ModelPulseTraceConfig") -> "PLModelLoader":
    """Return the appropriate model loader based on the model configuration."""
    model_loaders = {
        "tf": TensorFlowModelLoader,
        "keras": TensorFlowModelLoader,
        "pt": PyTorchModelLoader,
        "sklearn": SklearnModelLoader,
    }

    return create_component(model_config, "model", model_loaders)


def get_dataset_loader(dataset_config: "DatasetPulseTraceConfig") -> "PLDataLoader":
    """Return the appropriate dataset loader based on the dataset configuration."""
    dataset_loaders = {
        "csv": CSVDataLoader,
        "image": ImageDataLoader,
        "text": TextDataLoader,
    }

    return create_component(dataset_config, "dataset", dataset_loaders)


def get_explainer(explainer_config: "ExplainerPulseTraceConfig") -> "PLExplainer":
    """Return the appropriate explainer based on the explainer configuration."""
    explainers = {
        "lime": LimeExplainer,
        "shap": ShapExplainer,
    }

    return create_component(explainer_config, "explainer", explainers)


def run_global_explanation(
    model: "PLModel", explainer: "BaseExplainer", dataset: "PTDataSet"
) -> dict[ty.Literal["global_explanation"], ty.Any]:
    """Run global explanation on the model using the provided explainer and dataset."""
    pllogger.info("Running global explanation...")
    result = explainer.explain_global(model, dataset)
    pllogger.info("Global explanation completed.")

    return result


def run_local_explanation(
    model: "PLModel",
    explainer: "BaseExplainer",
    input_instance: "PTDataSet",
    dataset: "PTDataSet",
) -> dict[ty.Literal["local_explanation"], ty.Any]:
    """Run local explanation for the provided input instance."""
    pllogger.info("Running local explanation for the provided input instance...")
    result = explainer.explain_local(model, input_instance, dataset)
    pllogger.info("Local explanation completed.")

    return result


def load_model_and_data(config: dict) -> tuple:
    """Load model and dataset based on configuration."""
    # Load model
    model_config = config.get("model", {})
    model = get_model_loader(model_config).load_model()
    pllogger.info("Model loaded successfully.")

    # Load dataset
    dataset_config = config.get("dataset", {})
    dataset = get_dataset_loader(dataset_config).load_data(input_=False)
    pllogger.info("Dataset loaded successfully.")

    return model, dataset


def run_explanation(config: dict, model: "PLModel", dataset: "PTDataSet") -> dict:
    """Run explanation based on configuration mode."""
    # Load explainer
    explainer_config = config.get("explainer", {})
    explainer = get_explainer(explainer_config)

    # Run based on mode
    app_config = config.get("app", {})
    mode = app_config.get("mode", "global").lower()

    if mode == "global":
        return run_global_explanation(model, explainer, dataset)

    if mode == "local":
        local_config = config.get("local", {})
        input_instance = get_dataset_loader(local_config.get("dataset", {})).load_data(input_=True)

        return run_local_explanation(model, explainer, input_instance, dataset)

    msg = f"Invalid application mode: '{mode}'. Use 'global' or 'local'."
    raise ValueError(msg)


def main() -> None:
    """Entry point for the application."""
    try:
        # Parse arguments and load configuration
        args = parse_args()
        config = load_configuration(Path(args.cfg))

        # Setup logging
        logging_config = config.get("logging", {})
        setup_logging(logging_config)
        pllogger.info("Configuration loaded successfully.")

        # Load model and data
        model, dataset = load_model_and_data(config)

        # Run explanation
        results = run_explanation(config, model, dataset)

        # Log results
        pllogger.info("Explanation Results:")
        pllogger.info(results)

    except Exception as e:
        pllogger.exception(f"Error running application: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
