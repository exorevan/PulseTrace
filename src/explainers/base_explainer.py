import typing as ty
from abc import ABC, abstractmethod

if ty.TYPE_CHECKING:
    import numpy.typing as npt
    from pandas import DataFrame, Series


class BaseExplainer(ABC):
    config: dict[str, ty.Any]

    def __init__(self, config: dict[str, ty.Any]) -> None:
        self.config = config

    @abstractmethod
    def explain_global(
        self,
        model: ty.Any,  # e.g.: tf.keras.Model | torch.nn.Module | sklearn.base.BaseEstimator | etc.
        dataset: "DataFrame | npt.NDArray[ty.Any] | list[list[float]]",
    ) -> dict[str, ty.Any]:
        """
        Generate a global explanation for the provided dataset.
        :param model: The machine learning model.
                      Possible types: tf.keras.Model | torch.nn.Module | sklearn.base.BaseEstimator | etc.
        :param dataset: The dataset for which to generate explanations.
                        Example types: pandas.DataFrame | numpy.ndarray | list[list[float]].
        :return: A dictionary with global explanation results.
        """

        pass

    @abstractmethod
    def explain_local(
        self,
        model: ty.Any,  # e.g.: tf.keras.Model | torch.nn.Module | sklearn.base.BaseEstimator | etc.
        input_instance: "Series | list[float] | npt.NDArray[ty.Any]",
    ) -> dict[str, ty.Any]:
        """
        Generate a local explanation for a specific input instance.
        :param model: The machine learning model.
                      Possible types: tf.keras.Model | torch.nn.Module | sklearn.base.BaseEstimator | etc.
        :param input_instance: The input sample (or a one-row DataFrame/Series) to explain.
                               Example types: pandas.Series | list[float] | numpy.ndarray.
        :return: A dictionary with local explanation results.
        """

        pass
