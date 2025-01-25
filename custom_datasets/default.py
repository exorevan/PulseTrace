from pathlib import Path
import typing as ty

import numpy as np
import numpy.typing as npt


class DefaultDataset:
    """
    CSV dataset
    """

    X: npt.NDArray[npt.NDArray[np.float64]]
    y: npt.NDArray[np.float64]

    def __init__(self, path: Path, delimeter: str) -> None:
        data = np.loadtxt(path, delimiter=delimeter)

        self.X = np.array(data[:, :-1])
        self.y = np.array(data[:, -1])
