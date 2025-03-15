import importlib.util
import os
import typing as ty
from pathlib import Path

import torch

from .base_model_loader import BaseModelLoader


class PyTorchModelLoader(BaseModelLoader):
    @ty.override
    def load_model(self) -> ty.Any:
        # Load the architecture from a separate file
        arch_config = self.config.get("architecture")

        if not arch_config:
            raise Exception("Architecture configuration missing for PyTorch model.")

        arch_path = arch_config.get("path")
        class_name = arch_config.get("class_name")
        init_params = arch_config.get("init_params", {})

        if not arch_path or not Path.exists(arch_path):
            msg = f"Architecture file not found: {arch_path}"
            raise FileNotFoundError(msg)

        spec = importlib.util.spec_from_file_location("model_arch", arch_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, class_name):
            msg = f"Class {class_name} not found in module {arch_path}."
            raise Exception(msg)

        model_class = getattr(module, class_name)
        model = model_class(**init_params)

        # Load the weights
        weights_path = self.config.get("weights_path")
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            msg = f"Error loading PyTorch model weights from {weights_path}: {e}"
            raise Exception(msg) from e
        else:
            return model
