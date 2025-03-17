import typing as ty
from abc import ABC, abstractmethod
from collections import OrderedDict

if ty.TYPE_CHECKING:
    from pulsetrace.datasets_.base_data_loader import PTDataSet
    from pulsetrace.pltypes.config import ExplainerPulseTraceConfig


class BaseExplainer(ABC):
    config: "ExplainerPulseTraceConfig"

    def __init__(self, config: "ExplainerPulseTraceConfig") -> None:
        """
        Abstract base class for model explainability implementations.

        This class defines the interface for explainers that provide interpretability
        for machine learning models. Concrete implementations should extend this class
        to provide algorithm-specific explanation techniques.

        Attributes:
            config (ExplainerPulseTraceConfig): Configuration for the explainer.

        """
        self.config = config

    def find_matching_key(self, column: str, keys: list[str]) -> str | None:
        """
        Find a key in the provided list that contains the specified column.

        Args:
            column (str): The column name to search for.
            keys (list[str]): List of keys to search through.

        Returns:
            str | None: The first key that contains the column name, or None if no match is found.

        """
        for key in keys:
            if column in key:
                return key

        return None

    def sort_dict_by_columns(
        self, averaged_explanation: dict[str, float], dataset_columns: list[str]
    ) -> OrderedDict[str, float]:
        """
        Sort explanation dictionary based on the order of columns in the dataset.

        Args:
            averaged_explanation (dict[str, float]): Dictionary mapping feature names to importance scores.
            dataset_columns (list[str]): List of column names from the dataset in their original order.

        Returns:
            OrderedDict[str, float]: An ordered dictionary with explanation entries sorted by the
                order of columns in the dataset.

        """
        column_to_key = {}
        for column in dataset_columns:
            matching_key = self.find_matching_key(column, averaged_explanation.keys())
            if matching_key:
                column_to_key[column] = matching_key

        sorted_dict = OrderedDict()
        for column in dataset_columns:
            if column in column_to_key:
                feat = column_to_key[column]
                sorted_dict[feat] = averaged_explanation[feat]

        for feat, exaplanation in averaged_explanation.items():
            if feat not in sorted_dict:
                sorted_dict[feat] = exaplanation

        return sorted_dict

    @abstractmethod
    def explain_global(
        self,
        model: ty.Any,  # e.g.: tf.keras.Model | torch.nn.Module | sklearn.base.BaseEstimator | etc.
        dataset: "PTDataSet",
    ) -> dict[str, ty.Any]:
        """
        Generate global explanation for a model using LIME for tabular data.

        Creates a global explanation by averaging local explanations for multiple instances
        from the dataset, showing the overall feature importance for each class.

        Args:
            model (PLModel): The model to explain.
            dataset (PTDataSet): The dataset to use for explanation.

        Returns:
            dict[str, dict[int | str, OrderedDict[str, float]]]: A nested dictionary containing
                global explanations for each class, with feature importance scores.

        """

    @abstractmethod
    def explain_local(
        self, model: ty.Any, input_instance: "PTDataSet", dataset: "PTDataSet"
    ) -> dict[str, ty.Any]:
        """
        Generate local explanation for a specific instance using LIME for tabular data.

        Creates an explanation for a single instance, showing how each feature
        contributes to the model's prediction for that specific instance.

        Args:
            model (PLModel): The model to explain.
            input_instance (PTDataSet): The specific instance to explain.
            dataset (PTDataSet): The dataset used for reference.

        Returns:
            dict[str, dict[int | str, OrderedDict[str, float]]]: A nested dictionary containing
                local explanations with feature importance scores.

        """
