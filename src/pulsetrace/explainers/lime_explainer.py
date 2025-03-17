import typing as ty
from collections import OrderedDict

import numpy as np
from pulsetrace.logger import pllogger
from lime.lime_tabular import LimeTabularExplainer

from .base_explainer import BaseExplainer

if ty.TYPE_CHECKING:
    from lime.explanation import Explanation

    from pulsetrace.datasets_.base_data_loader import PTDataSet
    from pulsetrace.pltypes.config import ExplainerPulseTraceConfig
    from pulsetrace.pltypes.models import PLModel


@ty.final
class LimeExplainer(BaseExplainer):
    """
    Provide explainability for machine learning models using LIME for tabular data.

    This class implements the Local Interpretable Model-agnostic Explanations (LIME)
    algorithm for tabular data, supporting both global explanations (across multiple
    samples) and local explanations (for individual instances).
    """

    def __init__(self, config: "ExplainerPulseTraceConfig") -> None:
        """
        Initialize the LIME explainer with configuration parameters.

        Args:
            config (ExplainerPulseTraceConfig): Configuration object containing parameters
                for the explainer, such as num_features and num_samples.

        """
        super().__init__(config)
        self.num_features = config.get("parameters", {}).get("num_features", 10)
        self.num_samples = config.get("parameters", {}).get("num_samples", 5000)

    def _create_explainer(
        self, dataset: "PTDataSet", model: "PLModel", feature_names: list[int | str] | None = None
    ) -> LimeTabularExplainer:
        """
        Create a LimeTabularExplainer instance.

        Args:
            dataset (PTDataSet): The dataset to explain.
            model (PLModel): The model to explain.
            feature_names (list[str] | None): Names of features. If None, uses dataset.feature_names.
                Defaults to None.

        Returns:
            LimeTabularExplainer: A configured LIME tabular explainer instance.

        """
        names = feature_names or dataset.feature_names
        mode = "classification" if hasattr(model, "predict_proba") else "regression"

        return LimeTabularExplainer(
            dataset.data,
            feature_names=names,
            mode=mode,
            discretize_continuous=True,
        )

    def _get_prediction_function(self, model: "PLModel") -> ty.Callable[..., ty.Any]:
        """
        Get the appropriate prediction function from the model.

        Args:
            model (PLModel): The model to get the prediction function from.

        Returns:
            Callable[..., Any]: Either model.predict_proba (for classification) or
                model.predict (for regression).

        """
        return ty.cast(ty.Callable[..., ty.Any], model.predict_proba if hasattr(model, "predict_proba") else model.predict)

    @ty.override
    def explain_global(
        self, model: "PLModel", dataset: "PTDataSet"
    ) -> dict[str, dict[str, OrderedDict[str, float]]]:
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
        pllogger.info("Generating global explanation using LIME for tabular data...")

        mode = "classification" if hasattr(model, "predict_proba") else "regression"
        predict_fn = self._get_prediction_function(model)

        explainer = self._create_explainer(dataset, model)

        n_samples = min(10, len(dataset))
        classes = dataset.classes
        num_classes = len(classes)

        feature_totals = {label: {} for label in classes}
        feature_counts = {label: {} for label in classes}

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
                        feature_totals[label][feat] = weight
                        feature_counts[label][feat] = 1
                    else:
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
                str(label): self.sort_dict_by_columns(explanation, dataset.columns)
                for label, explanation in averaged_explanation.items()
            }

        return {"global_explanation": sorted_explanation}

    @ty.override
    def explain_local(
        self, model: "PLModel", input_instance: "PTDataSet", dataset: "PTDataSet"
    ) -> dict[str, dict[str, OrderedDict[str, float]]]:
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
        pllogger.info("Generating local explanation using LIME for tabular data...")

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
                str(dataset.classes[max_index]): self.sort_dict_by_columns(
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
