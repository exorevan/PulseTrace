import typing as ty
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from lime.lime_tabular import LimeTabularExplainer

from pulsetrace import custom_datasets
from pulsetrace.model_loader import SUPPORTED_MODELS_SKLEARN

if ty.TYPE_CHECKING:
    import numpy.typing as npt
    from lime.explanation import Explanation

    from pulsetrace.custom_datasets import DefaultDataset
    from pulsetrace.typings.config import PulseTraceConfig


DATATYPES = {"np.array": np.array, "np.ndarray": np.array}


class LimeExplanation:
    @classmethod
    def explain_sklearn(
        cls, model: SUPPORTED_MODELS_SKLEARN, config: "PulseTraceConfig"
    ) -> None:
        input_config = config["input"]
        dataset_config = config["dataset"]
        output_config = config["output"]

        input_data = DATATYPES[input_config["input_type"]](input_config["values"])

        feature_names = input_config["feature_names"]
        num_features = len(feature_names)
        class_names = input_config["class_names"]

        dataset_dict = dict(**dataset_config)

        dataset_type: str = str(dataset_dict.pop("type"))

        dataset_class: type | None = getattr(custom_datasets, dataset_type, None)

        if not dataset_class:
            raise TypeError(f"No dataset type {dataset_type}")

        dataset: "DefaultDataset" = dataset_class(**dataset_dict)

        explainer = LimeTabularExplainer(
            training_data=dataset.x,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True,
        )

        model_function = ty.cast(
            ty.Callable[[npt.NDArray[ty.Any]], npt.NDArray[ty.Any]],
            getattr(model, config["input"]["function"]),
        )

        explanation: "Explanation" = explainer.explain_instance(
            input_data,
            model_function,
            num_features=num_features,
            top_labels=1,
        )

        output_path = Path(output_config["path"])
        base_name = output_config["name"]

        html_file = output_path / f"{base_name}.html"

        if html_file.exists():
            dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            html_file = output_path / f"{base_name}_{dt_str}.html"

        explanation.save_to_file(html_file)
