import os
import typing as ty

import joblib

from .base_model_loader import BaseModelLoader


class SklearnModelLoader(BaseModelLoader):
    @ty.override
    def load_model(self) -> ty.Any:
        model_path: ty.Any = self.config.get("path")

        if not model_path:
            raise ValueError("scikit-learn model path not provided in configuration.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            model = joblib.load(model_path)

            return model
        except Exception as e:
            raise Exception(
                f"Error loading scikit-learn model from {model_path}: {e}"
            ) from e
