import typing as ty

import keras

from .base_model_loader import BaseModelLoader


class TensorFlowModelLoader(BaseModelLoader):
    @ty.override
    def load_model(self) -> keras.models.Model:
        model_path: ty.Any = self.config.get("path")

        if not model_path:
            raise ValueError("TensorFlow model path not provided in configuration.")

        try:
            model = ty.cast(keras.models.Model, keras.models.load_model(model_path))

            return model
        except Exception as e:
            msg = f"Error loading TensorFlow model from {model_path}: {e}"
            raise Exception(msg) from e
