from pathlib import Path

import joblib
from sklearn.svm import SVC

SUPPORTED_MODELS_SKLEARN = SVC

SUPPORTED_MODELS = SUPPORTED_MODELS_SKLEARN


class ModelLoader:
    @classmethod
    def init_sklearn(cls, path: Path) -> SUPPORTED_MODELS_SKLEARN:
        loaded_model: SUPPORTED_MODELS_SKLEARN = joblib.load(path)

        return loaded_model
