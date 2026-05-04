# PulseTrace

Explainable AI (XAI) command-line tool. Load a trained model and a dataset, then generate human-interpretable explanations using **LIME** or **SHAP** — globally across the full dataset or locally for a single instance.

---

## Installation

```bash
# Install uv (if not already installed)
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install project dependencies
uv sync
```

---

## Usage

```bash
uv run pulsetrace --cfg configs/<config>.yaml
```

To save output as a self-contained HTML page, set `output_format: html` in the config — the file is written automatically to `outputs/` and the path is printed to the console:

```bash
uv run pulsetrace --cfg configs/sklearn_iris_shap.yaml
# HTML saved → outputs/shap_global_classification_20260424_120000.html
```

**Note:** Image configs (`keras_mnist_*`, `keras_cifar10_*`, `keras_fashion_mnist_*`) require `output_format: html` or `output_format: json` — console output is not supported for image data.

---

## Examples by Modality

### Tabular — Scikit-learn models

**Iris (classification, built-in dataset)**

```bash
# Global — feature importance across the full dataset
uv run pulsetrace --cfg configs/sklearn_iris_lime.yaml
uv run pulsetrace --cfg configs/sklearn_iris_shap.yaml

# Local — explain a single instance
uv run pulsetrace --cfg configs/sklearn_iris_lime_local.yaml
uv run pulsetrace --cfg configs/sklearn_iris_shap_local.yaml
```

**House pricing (regression, CSV)**

```bash
# Global
uv run pulsetrace --cfg configs/sklearn_house_lime.yaml
uv run pulsetrace --cfg configs/sklearn_house_shap.yaml

# Local — explain one house (datasets/house_sample.csv)
uv run pulsetrace --cfg configs/sklearn_house_lime_local.yaml
uv run pulsetrace --cfg configs/sklearn_house_shap_local.yaml
```

**Alzheimer's disease (classification, CSV)**

```bash
# Global
uv run pulsetrace --cfg configs/sklearn_alzheimer_lime.yaml
uv run pulsetrace --cfg configs/sklearn_alzheimer_shap.yaml

# Local — explain one patient (datasets/alzheimer_sample.csv)
uv run pulsetrace --cfg configs/sklearn_alzheimer_lime_local.yaml
uv run pulsetrace --cfg configs/sklearn_alzheimer_shap_local.yaml
```

---

### Tabular — Keras neural network

**House pricing (regression, CSV) — Keras model, PyTorch backend**

```bash
# Global
uv run pulsetrace --cfg configs/keras_house_lime.yaml
uv run pulsetrace --cfg configs/keras_house_shap.yaml

# Local
uv run pulsetrace --cfg configs/keras_house_lime_local.yaml
uv run pulsetrace --cfg configs/keras_house_shap_local.yaml
```

---

### Image — Keras classifiers (PyTorch backend)

Weights must be trained before first use:

```bash
uv run python scripts/train_image_models.py
```

**MNIST — handwritten digits (28×28 grayscale, 10 classes)**

```bash
# Global
uv run pulsetrace --cfg configs/keras_mnist_lime.yaml
uv run pulsetrace --cfg configs/keras_mnist_shap.yaml

# Local — explain a single image
uv run pulsetrace --cfg configs/keras_mnist_lime_local.yaml
uv run pulsetrace --cfg configs/keras_mnist_shap_local.yaml
```

**CIFAR-10 — natural images (32×32 RGB, 10 classes)**

```bash
# Global
uv run pulsetrace --cfg configs/keras_cifar10_lime.yaml
uv run pulsetrace --cfg configs/keras_cifar10_shap.yaml

# Local
uv run pulsetrace --cfg configs/keras_cifar10_lime_local.yaml
uv run pulsetrace --cfg configs/keras_cifar10_shap_local.yaml
```

**Fashion-MNIST — clothing items (28×28 grayscale, 10 classes)**

```bash
# Global
uv run pulsetrace --cfg configs/keras_fashion_mnist_lime.yaml
uv run pulsetrace --cfg configs/keras_fashion_mnist_shap.yaml

# Local
uv run pulsetrace --cfg configs/keras_fashion_mnist_lime_local.yaml
uv run pulsetrace --cfg configs/keras_fashion_mnist_shap_local.yaml
```

---

## Config Reference

Key fields in every YAML config:

```yaml
app:
  mode: global | local          # global = full dataset; local = single instance
  output_format: console | json | html

model:
  type: sklearn | keras | pt
  path: weights/<file>

dataset:
  type: csv | builtin           # csv = local file; builtin = sklearn/keras dataset
  # csv options:
  path: datasets/<file>.csv
  delimiter: ","
  header: 0                     # null for headerless files
  only_x: true                  # true when file has no target column
  # builtin options:
  name: iris | mnist | cifar10 | ...

explainer:
  type: lime | shap
  num_features: 10

local:                          # required when mode: local
  dataset:
    type: csv
    path: datasets/<sample>.csv
    header: null
    only_x: true
```

See `configs/example.yaml` for a fully annotated reference config.

---

## Running Tests

```bash
uv run pytest -m "not slow" -q    # fast tests (~179 tests)
uv run pytest -q                   # all tests including slow (keras builtin datasets)
```
