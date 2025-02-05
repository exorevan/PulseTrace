import keras


class TensorFlowModelLoader:
    def __init__(self, config):
        self.config = config

    def load_model(self):
        model_path = self.config.get("path")
        if not model_path:
            raise ValueError("TensorFlow model path not provided in configuration.")
        try:
            model = keras.models.load_model(model_path)

            return model
        except Exception as e:
            raise Exception(f"Error loading TensorFlow model from {model_path}: {e}")
