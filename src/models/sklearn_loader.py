import os

import joblib


class SklearnModelLoader:
    def __init__(self, config):
        self.config = config

    def load_model(self):
        model_path = self.config.get("path")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)
        return model
