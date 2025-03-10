from collections import OrderedDict, defaultdict
import logging
import typing as ty

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from .base_explainer import BaseExplainer

if ty.TYPE_CHECKING:
    from lime.explanation import Explanation

    from datasets.base_data_loader import PTDataSet
    from pltypes.config import ExplainerPulseTraceConfig
    from pltypes.models import PLModel


@ty.final
class LimeExplainer(BaseExplainer):
    def __init__(self, config: "ExplainerPulseTraceConfig"):
        super().__init__(config)
        self.num_features = config.get("parameters", {}).get("num_features", 10)
        self.num_samples = config.get("parameters", {}).get("num_samples", 5000)

    @ty.override
    def explain_global(
        self, model: "PLModel", dataset: "PTDataSet"
    ) -> dict[str, dict[int | str, OrderedDict[str, float]]]:
        logging.info("Generating global explanation using LIME for tabular data...")

        mode = "classification" if hasattr(model, "predict_proba") else "regression"
        predict_fn: ty.Callable[..., ty.Any] = (
            model.predict_proba if mode == "classification" else model.predict
        )

        explainer = LimeTabularExplainer(
            dataset.get_x(),
            feature_names=dataset.feature_names,
            mode=mode,
            discretize_continuous=True,
        )

        n_samples = min(10, len(dataset))
        classes = dataset.classes
        num_classes = len(classes)
        aggregated_explanations = {label: defaultdict(list) for label in classes}

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
                explanation_list = ty.cast(
                    list[tuple[str, float]], explanation.as_list(label=idx)
                )

                for feat, weight in explanation_list:
                    aggregated_explanations[label][feat].append(weight)

        averaged_explanation: dict[str | int, dict[str, float]] = {
            label: {
                feat: sum(weights) / len(weights) for feat, weights in feats.items()
            }
            for label, feats in aggregated_explanations.items()
        }

        sorted_explanation = {
            label: self.sort_dict_by_columns(explanation, dataset.columns)
            for label, explanation in averaged_explanation.items()
        }

        return {"global_explanation": sorted_explanation}

    @ty.override
    def explain_local(
        self, model: "PLModel", input_instance: "PTDataSet", dataset: "PTDataSet"
    ) -> dict[str, dict[int | str, OrderedDict[str, float]]]:
        logging.info("Generating local explanation using LIME for tabular data...")

        if hasattr(dataset, "columns"):
            feature_names = list(dataset.columns)
            instance = input_instance.data
        else:
            instance = np.array(input_instance)
            feature_names = [f"f{i}" for i in range(len(instance))]

        mode = "classification" if hasattr(model, "predict_proba") else "regression"
        predict_fn: ty.Callable[..., ty.Any] = (
            model.predict_proba if mode == "classification" else model.predict
        )

        mode = "classification" if hasattr(model, "predict_proba") else "regression"
        explainer = LimeTabularExplainer(
            dataset.get_x(),
            feature_names=feature_names,
            mode=mode,
            discretize_continuous=True,
        )
        explanation = ty.cast(
            "Explanation",
            explainer.explain_instance(
                instance[0], predict_fn, num_features=self.num_features
            ),
        )

        predict = predict_fn([instance[0]])[0]

        max_index = np.argmax(predict)
        sorted_explanation = {
            dataset.classes[max_index]: self.sort_dict_by_columns(
                dict(explanation.as_list()), dataset.columns
            )
        }

        return {"local_explanation": sorted_explanation}
