"""Generate sktime benchmark time series datasets and train sklearn classifiers.

Datasets produced:
  - ItalyPowerDemand  (1096 samples, 24 timesteps, binary classification)
  - ArrowHead         ( 211 samples, 251 timesteps, 3-class classification)

Output files:
  datasets/ts_italy_power_train.npy  — (67, 25)   train split + label col
  datasets/ts_italy_power_test.npy   — (1029, 25)  test split  + label col
  datasets/ts_italy_power_sample.npy — (1, 24)     single sample for local mode
  datasets/ts_arrow_head_train.npy   — (36, 252)   train split + label col
  datasets/ts_arrow_head_test.npy    — (175, 252)  test split  + label col
  datasets/ts_arrow_head_sample.npy  — (1, 251)    single sample for local mode
  weights/ts_italy_mlp.pkl           — sklearn Pipeline (StandardScaler + MLP)
  weights/ts_arrow_mlp.pkl           — sklearn Pipeline (StandardScaler + MLP)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.modules.setdefault("tensorflow", None)


def _save_dataset(X: np.ndarray, y_enc: np.ndarray, path: Path) -> None:
    """Stack X and label column then save as .npy."""
    arr = np.hstack([X, y_enc.reshape(-1, 1)])
    np.save(path, arr)
    print(f"  saved {arr.shape} -> {path}")


def _save_sample(X: np.ndarray, path: Path, n: int = 1) -> None:
    """Save n rows of X without a target column (only_x=true)."""
    np.save(path, X[:n])
    print(f"  saved sample {X[:n].shape} -> {path}")


def _train_and_save(
    X: np.ndarray,
    y: np.ndarray,
    weights_path: Path,
    hidden_layers: tuple[int, ...] = (64, 32),
) -> None:
    """Train a StandardScaler + MLPClassifier pipeline and pickle it."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=500,
            random_state=42,
        )),
    ])
    model.fit(X, y)
    acc = model.score(X, y)
    print(f"  train acc: {acc:.3f}")
    with open(weights_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  weights -> {weights_path}")


def italy_power_demand() -> None:
    """ItalyPowerDemand: 24 timesteps, binary (classes 1/2 -> 0/1)."""
    print("\n=== ItalyPowerDemand ===")
    from sktime.datasets import load_italy_power_demand

    X_tr_raw, y_tr = load_italy_power_demand(split="train", return_type="numpy3D")
    X_te_raw, y_te = load_italy_power_demand(split="test", return_type="numpy3D")
    X_tr: np.ndarray = X_tr_raw[:, 0, :]   # (67, 24)
    X_te: np.ndarray = X_te_raw[:, 0, :]   # (1029, 24)

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr).astype(float)
    y_te_enc = le.transform(y_te).astype(float)
    print(f"  classes: {le.classes_} -> 0..{len(le.classes_) - 1}")

    _save_dataset(X_tr, y_tr_enc, Path("datasets/ts_italy_power_train.npy"))
    _save_dataset(X_te, y_te_enc, Path("datasets/ts_italy_power_test.npy"))
    _save_sample(X_te, Path("datasets/ts_italy_power_sample.npy"))

    X_all = np.vstack([X_tr, X_te])
    y_all = np.concatenate([y_tr_enc, y_te_enc])
    _train_and_save(X_all, y_all, Path("weights/ts_italy_mlp.pkl"))


def arrow_head() -> None:
    """ArrowHead: 251 timesteps, 3-class (classes 0/1/2)."""
    print("\n=== ArrowHead ===")
    from sktime.datasets import load_arrow_head

    X_tr_raw, y_tr = load_arrow_head(split="train", return_type="numpy3D")
    X_te_raw, y_te = load_arrow_head(split="test", return_type="numpy3D")
    X_tr: np.ndarray = X_tr_raw[:, 0, :]   # (36, 251)
    X_te: np.ndarray = X_te_raw[:, 0, :]   # (175, 251)

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr).astype(float)
    y_te_enc = le.transform(y_te).astype(float)
    print(f"  classes: {le.classes_} -> 0..{len(le.classes_) - 1}")

    _save_dataset(X_tr, y_tr_enc, Path("datasets/ts_arrow_head_train.npy"))
    _save_dataset(X_te, y_te_enc, Path("datasets/ts_arrow_head_test.npy"))
    _save_sample(X_te, Path("datasets/ts_arrow_head_sample.npy"))

    X_all = np.vstack([X_tr, X_te])
    y_all = np.concatenate([y_tr_enc, y_te_enc])
    _train_and_save(X_all, y_all, Path("weights/ts_arrow_mlp.pkl"), hidden_layers=(100, 64, 32))


if __name__ == "__main__":
    Path("datasets").mkdir(exist_ok=True)
    Path("weights").mkdir(exist_ok=True)
    italy_power_demand()
    arrow_head()
    print("\nAll done.")
