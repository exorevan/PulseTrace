import typing as ty
from abc import ABC, abstractmethod

if ty.TYPE_CHECKING:
    from pltypes.config import ModelPulseTraceConfig


class BaseModelLoader(ABC):
    config: "ModelPulseTraceConfig"

    def __init__(self, config: "ModelPulseTraceConfig") -> None:
        self.config = config

    @abstractmethod
    def load_model(self) -> ty.Any:
        """
        Load and return the machine learning model based on the configuration provided.

        :return: A machine learning model instance.

        """
        pass
