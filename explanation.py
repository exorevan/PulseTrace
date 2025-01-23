from datetime import datetime
import os
from pathlib import Path
import typing as ty

from lime.lime_tabular import LimeTabularExplainer
from lime.explanation import Explanation
import matplotlib.pyplot as plt
from model_loader import SUPPORTED_MODELS_SKLEARN
import numpy as np

if ty.TYPE_CHECKING:
    from config import PulseTraceConfig


DATATYPES = {"np.array": np.array, "np.ndarray": np.array}


class LimeExplanation:
    @classmethod
    def explain_sklearn(
        cls, model: SUPPORTED_MODELS_SKLEARN, config: "PulseTraceConfig"
    ) -> None:
        input_data = DATATYPES[config["input"]["input_type"]](config["input"]["values"])

        explainer = LimeTabularExplainer(
            training_data=input_data.reshape(1, -1),
            feature_names=config["input"][
                "feature_names"
            ],  # Adjust to your model’s features
            class_names=config["input"]["class_names"],  # Adjust to your actual classes
            discretize_continuous=True,
        )

        explanation: Explanation = explainer.explain_instance(
            input_data,
            ty.cast(ty.Callable, getattr(model, config["input"]["function"])),
            num_features=len(config["input"]["feature_names"]),
            top_labels=1,
        )

        os.makedirs(Path(config["output"]["path"]), exist_ok=True)

        html_file = os.path.join(
            Path(config["output"]["path"]), f"{config["output"]["name"]}.html"
        )

        if os.path.exists(html_file):
            dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            html_file = os.path.join(
                Path(config["output"]["path"]),
                f"{config['output']['name']}_{dt_str}.html",
            )

        explanation.save_to_file(html_file)
