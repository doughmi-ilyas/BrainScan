from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH = Path("models/brain_scan.keras")


def predict_image(image_path, model_path=MODEL_PATH):
    """Return the predicted class and confidence for one image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image cannot be read: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(image, verbose=0)[0]
    class_index = int(np.argmax(probabilities))

    return CLASS_NAMES[class_index], float(probabilities[class_index])


if __name__ == "__main__":
    image_path = input("Path of the MRI image: ").strip()
    predicted_class, confidence = predict_image(image_path)
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
