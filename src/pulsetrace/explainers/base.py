from __future__ import annotations

from abc import ABC, abstractmethod

from pulsetrace.adapters.base import ModelAdapter
from pulsetrace.config.schema import ExplainerConfig
from pulsetrace.data.dataset import Dataset

from .result import ExplanationResult


class BaseExplainer(ABC):
    def __init__(self, config: ExplainerConfig) -> None:
        self.config = config

    @abstractmethod
    def explain_global(self, model: ModelAdapter, dataset: Dataset) -> ExplanationResult:
        """Generate a global explanation: average feature importance over the dataset."""

    @abstractmethod
    def explain_local(
        self, model: ModelAdapter, instance: Dataset, dataset: Dataset
    ) -> ExplanationResult:
        """Generate a local explanation for instance.instance(0)."""
