from pathlib import Path

import numpy as np
import numpy.typing as npt


class DefaultDataset:
    """
    CSV dataset
    """

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    ids: npt.NDArray[np.int8] = np.array([])
    columns: npt.NDArray[np.str_] = np.array([])
    target_name: str = ""

    def __init__(
        self,
        path: Path,
        delimeter: str,
        id_column: bool = False,
        column_names: bool = False,
    ) -> None:
        data = np.loadtxt(path, object, delimiter=delimeter)

        if column_names:
            self.columns = np.array(data[0, :], np.str_)
            self.target_name = str(data[0][-1])

            data = data[1:]

        if id_column:
            self.ids = np.array(data[:, 0]).astype(int)

            data = data[:, 1:]

        self.x = np.array(data[:, :-1]).astype(float)
        self.y = np.array(data[:, -1])
