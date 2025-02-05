import os

from PIL import Image


class ImageDataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self, path=None):
        image_dir = path if path else self.config.get("path")
        if not image_dir or not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        images = []
        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                image_path = os.path.join(image_dir, filename)
                try:
                    img = Image.open(image_path)
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
        return images
