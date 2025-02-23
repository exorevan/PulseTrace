import logging
import typing as ty

import numpy as np
from lime.lime_tabular import LimeTabularExplainer


from .base_explainer import BaseExplainer

if ty.TYPE_CHECKING:
    from pltypes.config import ExplainerPulseTraceConfig
    from lime.explanation import Explanation
    from datasets.base_data_loader import PTDataSet


@ty.final
class LimeExplainer(BaseExplainer):
    def __init__(self, config: "ExplainerPulseTraceConfig"):
        super().__init__(config)
        self.num_features = config.get("parameters", {}).get("num_features", 10)
        self.num_samples = config.get("parameters", {}).get("num_samples", 5000)

    @ty.override
    def explain_global(
        self, model, dataset: "PTDataSet"
    ) -> dict[str, dict[str, float]]:
        logging.info("Generating global explanation using LIME for tabular data...")

        # Определяем режим работы на основе возможностей модели.
        mode = "classification" if hasattr(model, "predict_proba") else "regression"

        # Используем функцию предсказания, соответствующую режиму.
        predict_fn: ty.Callable[..., ty.Any] = (
            model.predict_proba
            if mode == "classification" and hasattr(model, "predict_proba")
            else model.predict
        )

        # Создаём LimeTabularExplainer без недопустимого параметра sample_size.
        explainer = LimeTabularExplainer(
            dataset.get_x(),
            feature_names=dataset.feature_names,
            mode=mode,
            discretize_continuous=True,
        )

        # Вычисляем локальные объяснения для части данных и агрегируем влияние признаков.

        n_samples = min(10, len(dataset))
        # Create a structure to hold explanations per class (assuming labels 0, 1, 2)
        aggregated_explanations = {label: {} for label in dataset.classes}

        for i in range(n_samples):
            instance = dataset[i]
            explanation = ty.cast(
                "Explanation",
                explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=self.num_features,
                    labels=range(len(dataset.classes)),
                ),
            )

            for idx, label in enumerate(dataset.classes):
                explanation_list = ty.cast(
                    list[tuple[str, float]], explanation.as_list(label=idx)
                )

                for feat, weight in explanation_list:
                    aggregated_explanations[label].setdefault(feat, []).append(weight)

        # Now, average the explanations for each class separately.
        averaged_explanation = {
            label: {
                feat: sum(weights) / len(weights) for feat, weights in feats.items()
            }
            for label, feats in aggregated_explanations.items()
        }

        return {"global_explanation": averaged_explanation}

    @ty.override
    def explain_local(self, model, input_instance: "PTDataSet"):
        logging.info("Generating local explanation using LIME for tabular data...")

        # Expect the input instance as a pandas DataFrame row or a 1D array.
        if hasattr(input_instance, "columns"):
            feature_names = list(input_instance.columns)
            instance = input_instance.data
            # In practice, background data should be a representative sample; here we use the input as fallback.
            background_data = input_instance.data
        else:
            instance = np.array(input_instance)
            feature_names = [f"f{i}" for i in range(len(instance))]
            background_data = np.array([instance])  # Fallback background data.

        mode = "classification" if hasattr(model, "predict_proba") else "regression"
        explainer = LimeTabularExplainer(
            background_data,
            feature_names=feature_names,
            mode=mode,
        )
        explanation = explainer.explain_instance(
            instance[0], model.predict, num_features=self.num_features
        )

        return {"local_explanation": dict(explanation.as_list())}
