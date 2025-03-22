from pulsetrace.datasets_.csv_loader import CSVDataLoader
from pulsetrace.datasets_.image_loader import ImageDataLoader
from pulsetrace.datasets_.text_loader import TextDataLoader
from pulsetrace.explainers.lime_explainer import LimeExplainer
from pulsetrace.explainers.shap_explainer import ShapExplainer
from pulsetrace.model_loaders.pytorch_loader import PyTorchModelLoader
from pulsetrace.model_loaders.sklearn_loader import SklearnModelLoader
from pulsetrace.model_loaders.tensorflow_loader import TensorFlowModelLoader


DATASET_LOADERS = {
    "csv": CSVDataLoader,
    "image": ImageDataLoader,
    "text": TextDataLoader,
}

MODEL_LOADERS = {
    "tf": TensorFlowModelLoader,
    "keras": TensorFlowModelLoader,
    "pt": PyTorchModelLoader,
    "sklearn": SklearnModelLoader,
}

EXPLAINERS = {
    "lime": LimeExplainer,
    "shap": ShapExplainer,
}
