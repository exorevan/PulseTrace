import argparse
import typing as ty
from pathlib import Path

import yaml
from datasets_ import CSVDataLoader, ImageDataLoader, TextDataLoader
from explainers import LimeExplainer, ShapExplainer
from logger import ptlogger
from model_loaders import PyTorchModelLoader, SklearnModelLoader, TensorFlowModelLoader
from pulsetrace.conf import DATASET_LOADERS, EXPLAINERS, MODEL_LOADERS
from utils.logger import setup_logging

if ty.TYPE_CHECKING:
    from src.pulsetrace.datasets_.base_data_loader import PTDataSet
    from src.pulsetrace.explainers.base_explainer import BaseExplainer
    from src.pulsetrace.pltypes import PLExplainer, PLModelLoader
    from pulsetrace.pltypes import PLDataLoader
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
        with cfg_path.open() as stream:
            return ty.cast("PulseTraceConfig", yaml.safe_load(stream))
    except FileNotFoundError:
        ptlogger.exception(f"Configuration file not found: {cfg_path}")
        raise
    except yaml.YAMLError as e:
        ptlogger.exception(f"Invalid YAML in configuration: {e}")
        raise
    except Exception as e:
        ptlogger.exception(f"Unexpected error loading configuration: {e}")
        raise


def create_component(
    config: dict[str, ty.Any], component_type: str, components_map: dict[str, ty.Any]
) -> ty.Any:
    """Create components based on configuration."""
    component_id = config.get("type", "").lower()
    component_class = components_map.get(component_id)

    if not component_class:
        valid_types = ", ".join(f"'{t}'" for t in components_map)
        msg = f"Unsupported {component_type} type: '{component_id}'. Valid types: {valid_types}."
        ptlogger.error(msg)
        raise ValueError(msg)

    return component_class(config)


def get_model_loader(model_config: "ModelPulseTraceConfig") -> "PLModelLoader":
    """Return the appropriate model loader based on the model configuration."""
    return create_component(model_config, "model", MODEL_LOADERS)


def get_dataset_loader(dataset_config: "DatasetPulseTraceConfig") -> "PLDataLoader":
    """Return the appropriate dataset loader based on the dataset configuration."""
    return create_component(dataset_config, "dataset", DATASET_LOADERS)


def get_explainer(explainer_config: "ExplainerPulseTraceConfig") -> "PLExplainer":
    """Return the appropriate explainer based on the explainer configuration."""
    return create_component(explainer_config, "explainer", EXPLAINERS)


def run_global_explanation(
    model: "PLModel", explainer: "BaseExplainer", dataset: "PTDataSet"
) -> dict[ty.Literal["global_explanation"], ty.Any]:
    """Run global explanation on the model using the provided explainer and dataset."""
    ptlogger.info("Running global explanation...")
    result = explainer.explain_global(model, dataset)
    ptlogger.info("Global explanation completed.")

    return result


def run_local_explanation(
    model: "PLModel",
    explainer: "BaseExplainer",
    input_instance: "PTDataSet",
    dataset: "PTDataSet",
) -> dict[ty.Literal["local_explanation"], ty.Any]:
    """Run local explanation for the provided input instance."""
    ptlogger.info("Running local explanation for the provided input instance...")
    result = explainer.explain_local(model, input_instance, dataset)
    ptlogger.info("Local explanation completed.")

    return result


def load_model_and_data(config: dict) -> tuple["PLModel", "PTDataSet"]:
    """Load model and dataset based on configuration."""
    # Load model
    model_config = config.get("model", {})
    model = get_model_loader(model_config).load_model()
    ptlogger.info("Model loaded successfully.")

    # Load dataset
    dataset_config = config.get("dataset", {})
    dataset = get_dataset_loader(dataset_config).load_data(input_=False)
    ptlogger.info("Dataset loaded successfully.")

    return model, dataset


def run_explanation(config: dict, model: "PLModel", dataset: "PTDataSet") -> dict:
    """Run explanation based on configuration mode."""
    # Load explainer
    explainer_config = config.get("explainer", {})
    explainer = get_explainer(explainer_config)

    # Run based on mode
    app_config = config.get("app", {})
    mode = app_config.get("mode", "global").lower()

    match mode:
        case "global":
            return run_global_explanation(model, explainer, dataset)
        case "local":
            local_config = config.get("local", {})
            input_instance = get_dataset_loader(
                local_config.get("dataset", {})
            ).load_data(input_=True)
            return run_local_explanation(model, explainer, input_instance, dataset)
        case _:
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
        ptlogger.info("Configuration loaded successfully.")

        # Load model and data
        model, dataset = load_model_and_data(config)

        # Run explanation
        results = run_explanation(config, model, dataset)

        # Log results
        ptlogger.info("Explanation Results:")
        ptlogger.info(results)

    except Exception as e:
        ptlogger.exception(f"Error running application: {e!s}")
        raise


if __name__ == "__main__":
    main()
