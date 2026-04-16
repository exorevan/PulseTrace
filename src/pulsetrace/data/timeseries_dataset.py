from __future__ import annotations

from dataclasses import dataclass

from .dataset import Dataset


@dataclass(frozen=True)
class TimeSeriesDataset(Dataset):
    """Dataset for time series. X has shape (n_samples, n_timesteps)."""

    @property
    def n_timesteps(self) -> int:
        return self.X.shape[1]
