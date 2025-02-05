import typing as ty
from abc import ABC, abstractmethod


class BaseModelLoader(ABC):
    def __init__(self, config: dict[str, ty.Any]) -> None:
        self.config = config

    @abstractmethod
    def load_model(self) -> ty.Any:
        """
        Load and return the machine learning model based on the configuration provided.
        :return: A machine learning model instance.
        """

        pass
