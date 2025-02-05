import logging
import typing as ty

import numpy as np
import shap

from .base_explainer import BaseExplainer

if ty.TYPE_CHECKING:
    import numpy.typing as npt
    from pandas import DataFrame, Series


class ShapExplainer(BaseExplainer):
    def __init__(self, config: dict[str, ty.Any]):
        super().__init__(config)

        self.num_features = config.get("parameters", {}).get("num_features", 10)

    @ty.override
    def explain_global(
        self,
        model: ty.Any,  # e.g.: tf.keras.Model | torch.nn.Module | sklearn.base.BaseEstimator | etc.
        dataset: "DataFrame | npt.NDArray[ty.Any] | list[list[float]]",
    ):
        logging.info("Generating global explanation using SHAP for tabular data...")

        # Determine feature names and convert dataset to NumPy array.
        if hasattr(dataset, "columns"):
            feature_names = list(dataset.columns)
            data = dataset.values
        else:
            data = np.array(dataset)
            feature_names = [f"f{i}" for i in range(data.shape[1])]

        # Select background data: use the first 100 rows if available.
        background = data[:100] if data.shape[0] > 100 else data

        # Create the SHAP explainer using the model and background data.
        # The explainer will automatically use the model's prediction function.
        explainer = shap.Explainer(model, background, feature_names=feature_names)

        # Select a sample subset for explanation purposes (e.g., first 10 rows).
        sample_data = data[:10]
        shap_values = explainer(sample_data)

        # Compute average absolute SHAP values for each feature to get a global measure.
        avg_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
        global_explanation = dict(zip(feature_names, avg_abs_shap_values))

        # Select the top features based on the computed average importance.
        sorted_explanation = dict(
            sorted(global_explanation.items(), key=lambda x: x[1], reverse=True)[
                : self.num_features
            ]
        )

        return {"global_explanation": sorted_explanation}

    @ty.override
    def explain_local(
        self,
        model: ty.Any,  # e.g.: tf.keras.Model | torch.nn.Module | sklearn.base.BaseEstimator | etc.
        input_instance: "Series | list[float] | npt.NDArray[ty.Any]",
    ):
        logging.info("Generating local explanation using SHAP for tabular data...")

        # Handle input instance as a pandas DataFrame row or a numpy array.
        if hasattr(input_instance, "columns"):
            feature_names = list(input_instance.columns)
            instance = input_instance.iloc[0].values.reshape(1, -1)
        else:
            instance = np.array(input_instance)
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)
            feature_names = [f"f{i}" for i in range(instance.shape[1])]

        # Use the instance itself as background if no better background is provided.
        background = instance

        # Create the SHAP explainer for the local instance.
        explainer = shap.Explainer(model, background, feature_names=feature_names)
        shap_values = explainer(instance)

        # Extract local SHAP values for the instance.
        local_explanation = dict(zip(feature_names, shap_values.values[0]))
        sorted_local_explanation = dict(
            sorted(local_explanation.items(), key=lambda x: abs(x[1]), reverse=True)[
                : self.num_features
            ]
        )

        return {"local_explanation": sorted_local_explanation}
