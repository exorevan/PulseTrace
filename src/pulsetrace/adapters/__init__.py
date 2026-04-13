from __future__ import annotations

from pulsetrace.config.schema import (
    KerasModelConfig,
    ModelConfig,
    SklearnModelConfig,
    TorchModelConfig,
)

from .base import ModelAdapter
from .keras import KerasAdapter
from .pytorch import PyTorchAdapter
from .sklearn import SklearnAdapter


def build_adapter(config: ModelConfig) -> ModelAdapter:
    """Load the model from disk and wrap it in the appropriate adapter."""
    if isinstance(config, SklearnModelConfig):
        import joblib

        model = joblib.load(config.path)
        return SklearnAdapter(model)

    if isinstance(config, KerasModelConfig):
        import keras

        model = keras.models.load_model(config.path)
        return KerasAdapter(model)

    if isinstance(config, TorchModelConfig):
        import importlib.util

        import torch

        spec = importlib.util.spec_from_file_location(
            "model_arch", config.architecture.path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load architecture from {config.architecture.path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        if not hasattr(module, config.architecture.class_name):
            raise AttributeError(
                f"Class '{config.architecture.class_name}' not found in {config.architecture.path}"
            )

        model_class = getattr(module, config.architecture.class_name)
        model = model_class(**config.architecture.init_params)

        state_dict = torch.load(config.weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        return PyTorchAdapter(model, task=config.task)

    raise ValueError(f"Unhandled model config type: {type(config)}")


__all__ = [
    "ModelAdapter",
    "SklearnAdapter",
    "KerasAdapter",
    "PyTorchAdapter",
    "build_adapter",
]
