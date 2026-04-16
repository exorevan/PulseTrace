import pytest
from pathlib import Path
from pydantic import ValidationError
from pulsetrace.config import load_config
from pulsetrace.config.schema import (
    PulseTraceConfig,
    SklearnModelConfig,
    KerasModelConfig,
    TorchModelConfig,
)


def test_valid_sklearn_global_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
model:
  type: sklearn
  path: weights/model.pkl
dataset:
  type: csv
  path: data.csv
explainer:
  type: lime
""")
    config = load_config(cfg)
    assert isinstance(config.model, SklearnModelConfig)
    assert config.explainer.type == "lime"
    assert config.app.mode == "global"
    assert config.explainer.num_features == 10
    assert config.explainer.num_samples == 5000


def test_valid_keras_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
model:
  type: keras
  path: weights/model.keras
dataset:
  type: csv
  path: data.csv
explainer:
  type: shap
  num_features: 5
""")
    config = load_config(cfg)
    assert isinstance(config.model, KerasModelConfig)
    assert config.explainer.num_features == 5


def test_valid_tf_alias(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
model:
  type: tf
  path: weights/model.keras
dataset:
  type: csv
  path: data.csv
explainer:
  type: lime
""")
    config = load_config(cfg)
    assert isinstance(config.model, KerasModelConfig)


def test_valid_local_mode_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
app:
  mode: local
model:
  type: sklearn
  path: weights/model.pkl
dataset:
  type: csv
  path: data.csv
explainer:
  type: lime
local:
  dataset:
    type: csv
    path: input.csv
    only_x: true
""")
    config = load_config(cfg)
    assert config.app.mode == "local"
    assert config.local is not None
    assert config.local.dataset.only_x is True


def test_unknown_model_type_raises(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
model:
  type: xgboost
  path: weights/model.pkl
dataset:
  type: csv
  path: data.csv
explainer:
  type: lime
""")
    with pytest.raises(ValidationError):
        load_config(cfg)


def test_missing_model_field_raises(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
dataset:
  type: csv
  path: data.csv
explainer:
  type: lime
""")
    with pytest.raises(ValidationError):
        load_config(cfg)


def test_invalid_explainer_type_raises(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
model:
  type: sklearn
  path: weights/model.pkl
dataset:
  type: csv
  path: data.csv
explainer:
  type: integrated_gradients
""")
    with pytest.raises(ValidationError):
        load_config(cfg)


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_config(Path("does_not_exist.yaml"))


def test_valid_torch_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
model:
  type: pt
  weights_path: weights/model.pt
  architecture:
    path: models/arch.py
    class_name: MyModel
    init_params:
      hidden_size: 128
  task: classification
dataset:
  type: csv
  path: data.csv
explainer:
  type: lime
""")
    config = load_config(cfg)
    assert isinstance(config.model, TorchModelConfig)
    assert config.model.task == "classification"
    assert config.model.architecture.init_params == {"hidden_size": 128}


def test_invalid_yaml_raises(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(": invalid: [yaml")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(cfg)


def test_empty_config_raises(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_config(cfg)


# ---------------------------------------------------------------------------
# TimeSeriesDatasetConfig + n_segments
# ---------------------------------------------------------------------------

from pulsetrace.config.schema import TimeSeriesDatasetConfig, ExplainerConfig


def test_timeseries_dataset_config_parses():
    cfg = TimeSeriesDatasetConfig(type="timeseries", path="datasets/my.npy")
    assert cfg.type == "timeseries"
    assert cfg.only_x is False
    assert cfg.n_timesteps is None
    assert cfg.target_col is None


def test_timeseries_dataset_config_only_x():
    cfg = TimeSeriesDatasetConfig(type="timeseries", path="datasets/my.npy", only_x=True)
    assert cfg.only_x is True


def test_explainer_config_has_n_segments():
    cfg = ExplainerConfig(type="lime", num_features=5)
    assert cfg.n_segments == 10  # default


def test_explainer_config_custom_n_segments():
    cfg = ExplainerConfig(type="lime", num_features=5, n_segments=20)
    assert cfg.n_segments == 20


def test_load_config_with_timeseries_dataset(tmp_path):
    import yaml
    cfg_path = tmp_path / "ts.yaml"
    cfg_path.write_text(yaml.dump({
        "model": {"type": "sklearn", "path": "weights/m.pkl"},
        "dataset": {"type": "timeseries", "path": "datasets/my.npy"},
        "explainer": {"type": "shap", "num_features": 5},
    }))
    cfg = load_config(cfg_path)
    assert cfg.dataset.type == "timeseries"
