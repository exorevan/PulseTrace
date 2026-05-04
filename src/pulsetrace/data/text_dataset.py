"""TextDataset — dataset subclass for raw text samples."""
from __future__ import annotations

from dataclasses import dataclass, field

from .dataset import Dataset


@dataclass(frozen=True)
class TextDataset(Dataset):
    """Dataset for text classification.

    texts[i] is the raw string corresponding to X[i].
    X is a dummy zeros array (n, 1); explainers use texts directly.
    """

    texts: list[str] = field(default_factory=list, kw_only=True)
