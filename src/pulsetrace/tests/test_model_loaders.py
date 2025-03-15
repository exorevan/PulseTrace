import importlib.util
import os

import joblib
import keras
import torch

from model_loaders.pytorch_loader import PyTorchModelLoader
from model_loaders.sklearn_loader import SklearnModelLoader
from model_loaders.tensorflow_loader import TensorFlowModelLoader
from pltypes import ModuleType


# Dummy model for TensorFlow and scikit-learn.
class DummyModel:
    pass


# Dummy PyTorch model class for testing.
class DummyPTModel:
    def __init__(self, **kwargs):
        self.params = kwargs

    def load_state_dict(self, state):
        self.state_dict = state

    def eval(self):
        self.evaluated = True


def test_tensorflow_model_loader(monkeypatch):
    dummy_model = DummyModel()

    def dummy_load_model(path):
        return dummy_model

    monkeypatch.setattr(keras.models, "load_model", dummy_load_model)
    config = {"path": "dummy_tf_model.h5"}
    tf_loader = TensorFlowModelLoader(config)
    model = tf_loader.load_model()
    assert model is dummy_model


def test_sklearn_model_loader(monkeypatch):
    dummy_model = DummyModel()

    def dummy_joblib_load(path):
        return dummy_model

    monkeypatch.setattr(joblib, "load", dummy_joblib_load)
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    config = {"path": "dummy_sklearn_model.pkl"}
    sk_loader = SklearnModelLoader(config)
    model = sk_loader.load_model()
    assert model is dummy_model


def test_pytorch_model_loader(monkeypatch):
    dummy_state = {"dummy": 1}

    # Create a dummy module that contains DummyPTModel.
    dummy_module = ModuleType("dummy_module")
    setattr(dummy_module, "DummyPTModel", DummyPTModel)

    # Create a dummy spec and loader.
    class DummyLoader:
        def exec_module(self, module):
            module.__dict__.update(dummy_module.__dict__)

    dummy_spec = type("DummySpec", (), {"loader": DummyLoader()})
    monkeypatch.setattr(
        importlib.util, "spec_from_file_location", lambda name, path: dummy_spec
    )
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    monkeypatch.setattr(torch, "load", lambda path, map_location: dummy_state)

    config = {
        "architecture": {
            "path": "dummy_arch.py",
            "class_name": "DummyPTModel",
            "init_params": {"param": "value"},
        },
        "weights_path": "dummy_weights.pt",
    }
    pt_loader = PyTorchModelLoader(config)
    model = pt_loader.load_model()
    assert isinstance(model, DummyPTModel)
    assert model.params == {"param": "value"}
    assert hasattr(model, "state_dict")
    assert model.state_dict == dummy_state
    assert hasattr(model, "evaluated")
    assert model.evaluated is True
