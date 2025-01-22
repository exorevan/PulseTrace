from types import ModuleType

import numpy as np
import sklearn
import tensorflow

from explaination import LimeExplaination
from model_loader import ModelLoader

MODEL_MODULE: dict[str, ModuleType] = {
    "sklearn": sklearn,
    "scikit-learn": sklearn,
    "tfflow": tensorflow,
    "tensorflow": tensorflow,
}

MODEL_LOADER_ACCORDANCE = {sklearn: ModelLoader.init_sklearn}

MODEL_EXPLAINATION_ACCORDANCE = {"lime": {sklearn: LimeExplaination.explain_sklearn}}
