from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def explain_global(self, model, dataset):
        """
        Generate a global explanation for the provided dataset.
        :param model: The machine learning model.
        :param dataset: The dataset for which to generate explanations.
        :return: A dictionary with global explanation results.
        """
        pass

    @abstractmethod
    def explain_local(self, model, input_instance):
        """
        Generate a local explanation for a specific input instance.
        :param model: The machine learning model.
        :param input_instance: The input sample (or a DataFrame with one row) to explain.
        :return: A dictionary with local explanation results.
        """
        pass
