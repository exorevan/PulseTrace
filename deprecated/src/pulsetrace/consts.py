from types import ModuleType

import sklearn
import tensorflow

from pulsetrace.explanation import LimeExplanation
from pulsetrace.model_loader import ModelLoader

MODEL_MODULE: dict[str, ModuleType] = {
    "sklearn": sklearn,
    "scikit-learn": sklearn,
    "tfflow": tensorflow,
    "tensorflow": tensorflow,
}

MODEL_LOADER_ACCORDANCE = {sklearn: ModelLoader.init_sklearn}

MODEL_EXPLANATION_ACCORDANCE = {"lime": {sklearn: LimeExplanation.explain_sklearn}}
