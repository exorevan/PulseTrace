"""Train and save minimal Keras image classifiers for the three builtin image datasets.

Run from the project root (requires a working TF/Keras installation):

    uv run python scripts/train_image_models.py

Saves to:
    weights/mnist_mlp.keras
    weights/cifar10_mlp.keras
    weights/fashion_mnist_mlp.keras
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.modules.setdefault("tensorflow", None)  # type: ignore[arg-type]  # Block broken TF

import importlib

import numpy as np


def train_and_save(dataset_name: str, out_path: str, n_train: int = 10_000, epochs: int = 5) -> None:
    import keras

    mod = importlib.import_module(f"keras.datasets.{dataset_name}")
    (x_train, y_train), _ = mod.load_data()

    x = x_train[:n_train].astype(np.float32) / 255.0
    if x.ndim == 3:
        x = np.expand_dims(x, axis=-1)  # (n, H, W) → (n, H, W, 1)
    y = y_train[:n_train].ravel()

    input_shape = x.shape[1:]

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x, y, epochs=epochs, batch_size=256, validation_split=0.1, verbose=1)
    model.save(out_path)

    preds = model.predict(x[:3], verbose=0)
    print(f"  {dataset_name}: saved → {out_path}")
    print(f"    input shape: {input_shape}  |  output shape: {preds.shape}  |  row[0] sum: {preds[0].sum():.4f}")


if __name__ == "__main__":
    datasets = [
        ("mnist", "weights/mnist_mlp.keras"),
        ("cifar10", "weights/cifar10_mlp.keras"),
        ("fashion_mnist", "weights/fashion_mnist_mlp.keras"),
    ]
    for name, path in datasets:
        print(f"\nTraining {name}...")
        train_and_save(name, path)

    print("\nAll models saved. Run the configs with:")
    for name, _ in datasets:
        print(f"  uv run pulsetrace --cfg configs/keras_{name}_lime.yaml")
