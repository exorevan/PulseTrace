import logging

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from .base_explainer import BaseExplainer


class LimeExplainer(BaseExplainer):
    def __init__(self, config):
        super().__init__(config)
        self.num_features = config.get("parameters", {}).get("num_features", 10)
        self.num_samples = config.get("parameters", {}).get("num_samples", 5000)

    def explain_global(self, model, dataset):
        logging.info("Generating global explanation using LIME for tabular data...")

        # Определяем режим работы на основе возможностей модели.
        mode = "classification" if hasattr(model, "predict_proba") else "regression"
        # Используем функцию предсказания, соответствующую режиму.
        predict_fn = (
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
        aggregated_explanations = {}

        for i in range(n_samples):
            instance = dataset[i]
            explanation = explainer.explain_instance(
                instance, predict_fn, num_features=self.num_features
            )
            for feat, weight in explanation.as_list():
                aggregated_explanations.setdefault(feat, []).append(weight)

        averaged_explanation = {
            feat: sum(weights) / len(weights)
            for feat, weights in aggregated_explanations.items()
        }

        return {"global_explanation": averaged_explanation}

    def explain_local(self, model, input_instance):
        logging.info("Generating local explanation using LIME for tabular data...")
        # Expect the input instance as a pandas DataFrame row or a 1D array.
        if hasattr(input_instance, "columns"):
            feature_names = list(input_instance.columns)
            instance = input_instance.iloc[0].values
            # In practice, background data should be a representative sample; here we use the input as fallback.
            background_data = input_instance.values
        else:
            instance = np.array(input_instance)
            feature_names = [f"f{i}" for i in range(len(instance))]
            background_data = np.array([instance])  # Fallback background data.

        mode = "classification" if hasattr(model, "predict_proba") else "regression"
        explainer = LimeTabularExplainer(
            background_data,
            feature_names=feature_names,
            mode=mode,
            sample_size=self.num_samples,
        )
        explanation = explainer.explain_instance(
            instance, model.predict, num_features=self.num_features
        )
        return {"local_explanation": dict(explanation.as_list())}
