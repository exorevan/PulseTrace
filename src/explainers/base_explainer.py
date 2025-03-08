from collections import OrderedDict
import typing as ty
from abc import ABC, abstractmethod

if ty.TYPE_CHECKING:
    from datasets.base_data_loader import PTDataSet
    from pltypes.config import ExplainerPulseTraceConfig


class BaseExplainer(ABC):
    config: "ExplainerPulseTraceConfig"

    def __init__(self, config: "ExplainerPulseTraceConfig") -> None:
        self.config = config

    def find_matching_key(self, column: str, keys: list[str]) -> str | None:
        for key in keys:
            if column in key:
                return key

        return None

    def sort_dict_by_columns(
        self, averaged_explanation: dict[str, float], dataset_columns: list[str]
    ) -> OrderedDict[str, float]:
        column_to_key = {}
        for column in dataset_columns:
            matching_key = self.find_matching_key(column, averaged_explanation.keys())
            if matching_key:
                column_to_key[column] = matching_key

        sorted_dict = OrderedDict()
        for column in dataset_columns:
            if column in column_to_key:
                key = column_to_key[column]
                sorted_dict[key] = averaged_explanation[key]

        for key in averaged_explanation:
            if key not in sorted_dict:
                sorted_dict[key] = averaged_explanation[key]

        return sorted_dict

    @abstractmethod
    def explain_global(
        self,
        model: ty.Any,  # e.g.: tf.keras.Model | torch.nn.Module | sklearn.base.BaseEstimator | etc.
        dataset: "PTDataSet",
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
        self, model: ty.Any, input_instance: "PTDataSet", dataset: "PTDataSet"
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
