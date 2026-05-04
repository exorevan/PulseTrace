"""Retrain house price regression model with Keras + PyTorch backend.

Uses the sklearn model's predictions as pseudo-labels (the CSV has no target column).

Run from the project root:

    uv run python scripts/retrain_keras_house.py

Overwrites:
    weights/house_lin_reg.keras
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.modules.setdefault("tensorflow", None)  # type: ignore[arg-type]  # Block broken TF

from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent


def main() -> None:
    # ── Load features ────────────────────────────────────────────────────────
    df = pd.read_csv(ROOT / "datasets" / "house_pricing_dataset.csv")
    X = df.values.astype(np.float32)
    n_features = X.shape[1]
    print(f"Loaded {X.shape[0]} samples, {n_features} features.")

    # ── Generate pseudo-labels from sklearn model ────────────────────────────
    sklearn_model = joblib.load(ROOT / "weights" / "house_lin_reg.pkl")
    y = sklearn_model.predict(df).astype(np.float32)
    print(f"Pseudo-label range: [{y.min():.1f}, {y.max():.1f}]")

    # ── Build keras regression model ─────────────────────────────────────────
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X, y, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = ROOT / "weights" / "house_lin_reg.keras"
    model.save(str(out_path))
    print(f"\nSaved → {out_path}")

    # ── Sanity check ─────────────────────────────────────────────────────────
    preds = model.predict(X[:3], verbose=0).flatten()
    ref = y[:3]
    print(f"Sample keras preds:  {preds}")
    print(f"sklearn reference:   {ref}")


if __name__ == "__main__":
    main()
