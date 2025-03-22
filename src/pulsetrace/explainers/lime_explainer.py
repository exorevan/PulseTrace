import logging
import typing as ty
from collections import OrderedDict, defaultdict

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from .base_explainer import BaseExplainer
from pulsetrace.logger import ptlogger

if ty.TYPE_CHECKING:
    from lime.explanation import Explanation

    from pulsetrace.datasets_.base_data_loader import PTDataSet
    from pulsetrace.pltypes.config import ExplainerPulseTraceConfig
    from pulsetrace.pltypes.models import PLModel


@ty.final
class LimeExplainer(BaseExplainer):
    """LIME explainer for tabular data."""

    def __init__(self, config: "ExplainerPulseTraceConfig") -> None:
        super().__init__(config)
        self.num_features = config.get("parameters", {}).get("num_features", 10)
        self.num_samples = config.get("parameters", {}).get("num_samples", 5000)

    def _create_explainer(
        self, dataset: "PTDataSet", model: "PLModel", feature_names=None
    ) -> LimeTabularExplainer:
        """Create a LIME tabular explainer with appropriate configuration."""
        names = feature_names or dataset.feature_names
        mode = "classification" if hasattr(model, "predict_proba") else "regression"

        return LimeTabularExplainer(
            dataset.data,
            feature_names=names,
            mode=mode,
            discretize_continuous=True,
        )

    def _get_prediction_function(self, model: "PLModel") -> ty.Callable[..., ty.Any]:
        """Get the appropriate prediction function based on model type."""
        return model.predict_proba if hasattr(model, "predict_proba") else model.predict

    @ty.override
    def explain_global(
        self, model: "PLModel", dataset: "PTDataSet"
    ) -> dict[str, dict[int | str, OrderedDict[str, float]]]:
        """Generate global explanation using LIME for tabular data."""
        ptlogger.info("Generating global explanation using LIME for tabular data...")

        mode = "classification" if hasattr(model, "predict_proba") else "regression"
        predict_fn = self._get_prediction_function(model)

        explainer = self._create_explainer(dataset, model)

        n_samples = min(10, len(dataset))
        classes = dataset.classes
        num_classes = len(classes)

        feature_totals = {label: defaultdict(float) for label in classes}
        feature_counts = {label: defaultdict(int) for label in classes}

        for i in range(n_samples):
            instance = dataset[i]
            explanation = ty.cast(
                "Explanation",
                explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=self.num_features,
                    labels=range(num_classes),
                ),
            )

            for idx, label in enumerate(classes):
                for feat, weight in explanation.as_list(label=idx):
                    if feat not in feature_totals[label]:
                        feature_totals[label][feat] += weight
                        feature_counts[label][feat] += 1

        averaged_explanation: dict[str | int, dict[str, float]] = {
            label: {
                feat: feature_totals[label][feat] / feature_counts[label][feat]
                for feat in feature_totals[label]
            }
            for label in classes
        }

        if mode == "regression":
            sorted_explanation = {
                dataset.target_name: self.sort_dict_by_columns(
                    next(iter(averaged_explanation.values())), dataset.columns
                )
            }
        else:
            sorted_explanation = {
                label: self.sort_dict_by_columns(explanation, dataset.columns)
                for label, explanation in averaged_explanation.items()
            }

        return {"global_explanation": sorted_explanation}

    @ty.override
    def explain_local(
        self, model: "PLModel", input_instance: "PTDataSet", dataset: "PTDataSet"
    ) -> dict[str, dict[int | str, OrderedDict[str, float]]]:
        """
        Generate local explanation for a single instance using LIME.

        Explains which features most influenced the prediction for a specific instance.

        """
        ptlogger.info("Generating local explanation using LIME for tabular data...")

        if hasattr(dataset, "columns"):
            feature_names = list(dataset.columns)
            instance = input_instance.data
        else:
            instance = np.array(input_instance)
            feature_names = [f"f{i}" for i in range(len(instance))]

        mode = "classification" if hasattr(model, "predict_proba") else "regression"
        predict_fn = self._get_prediction_function(model)

        mode = "classification" if hasattr(model, "predict_proba") else "regression"
        explainer = self._create_explainer(dataset, model, feature_names)
        explanation = ty.cast(
            "Explanation",
            explainer.explain_instance(
                instance[0], predict_fn, num_features=self.num_features
            ),
        )

        predict = predict_fn([instance[0]])[0]

        if mode == "classification":
            max_index = np.argmax(predict)
            sorted_explanation = {
                dataset.classes[max_index]: self.sort_dict_by_columns(
                    dict(explanation.as_list()), dataset.columns
                )
            }
        else:
            sorted_explanation = {
                dataset.target_name: self.sort_dict_by_columns(
                    dict(explanation.as_list()), dataset.columns
                )
            }

        return {
            "local_explanation": {
                dataset.target_name: next(iter(sorted_explanation.values()))
            }
        }
