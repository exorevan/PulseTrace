"""Download SST-2 samples and save as .txt files for PulseTrace text configs.

The Hub model "distilbert-base-uncased-finetuned-sst-2-english" is used directly
without retraining — this script only generates the sample dataset files.

Usage:
    uv run python scripts/generate_text_datasets.py

Creates:
    datasets/sst2_sample/positive/   10 positive-sentiment sentences
    datasets/sst2_sample/negative/   10 negative-sentiment sentences
    datasets/sst2_local/             1 sentence (for local mode configs)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.modules.setdefault("tensorflow", None)


def main() -> None:
    """Download SST-2 validation split and write sample .txt files."""
    from datasets import load_dataset  # type: ignore[import]

    print("Downloading SST-2 validation split...")
    ds = load_dataset("glue", "sst2", split="validation")

    pos_dir = Path("datasets/sst2_sample/positive")
    neg_dir = Path("datasets/sst2_sample/negative")
    local_dir = Path("datasets/sst2_local")

    for d in (pos_dir, neg_dir, local_dir):
        d.mkdir(parents=True, exist_ok=True)

    pos_samples = [ex for ex in ds if ex["label"] == 1][:10]
    neg_samples = [ex for ex in ds if ex["label"] == 0][:10]

    for i, ex in enumerate(pos_samples):
        (pos_dir / f"sample_{i:03d}.txt").write_text(ex["sentence"], encoding="utf-8")

    for i, ex in enumerate(neg_samples):
        (neg_dir / f"sample_{i:03d}.txt").write_text(ex["sentence"], encoding="utf-8")

    (local_dir / "sample_000.txt").write_text(neg_samples[0]["sentence"], encoding="utf-8")

    print(f"Saved {len(pos_samples)} positive samples → {pos_dir}")
    print(f"Saved {len(neg_samples)} negative samples → {neg_dir}")
    print(f"Saved 1 local sample → {local_dir}")
    print("\nRun the configs with:")
    print("  uv run pulsetrace text_sst2_lime")
    print("  uv run pulsetrace text_sst2_shap")


if __name__ == "__main__":
    main()
