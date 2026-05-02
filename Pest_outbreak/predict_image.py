# Pest_outbreak/predict_image.py
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib

import constants as constants

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use absolute paths
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_pest_model.h5")
META_PATH = os.path.join(SCRIPT_DIR, "paddy_meta.joblib")
CLASS_INDICES_PATH = os.path.join(SCRIPT_DIR, "class_indices.json")
DATASET_PATH = os.path.join(SCRIPT_DIR, "PestDataset")
NON_PEST_LABELS = {"none", "not_pest", "non_pest"}


def _dataset_class_names(dataset_path):
    class_names = []
    if not os.path.isdir(dataset_path):
        return class_names

    for entry in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, entry)
        if not os.path.isdir(class_path):
            continue

        has_image = any(
            os.path.splitext(filename)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
            for filename in os.listdir(class_path)
        )
        if has_image:
            class_names.append(entry)

    return class_names

def load_artifacts():
    """Load model and metadata"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata not found at: {META_PATH}")
    
    model = load_model(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta


def load_class_labels():
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as file_handle:
            class_indices = json.load(file_handle)
        return [label for label, _ in sorted(class_indices.items(), key=lambda item: item[1])]

    dataset_labels = _dataset_class_names(DATASET_PATH)
    if dataset_labels:
        return dataset_labels

    raise FileNotFoundError(
        f"Could not load class labels from {CLASS_INDICES_PATH} or dataset folder {DATASET_PATH}."
    )


def _non_pest_index(class_labels):
    for index, label in enumerate(class_labels):
        if label.lower() in NON_PEST_LABELS:
            return index
    return None


def _should_reject_prediction(predictions, predicted_index, class_labels):
    if not class_labels or predicted_index >= len(class_labels):
        return False

    predicted_label = class_labels[predicted_index]
    if predicted_label.lower() in NON_PEST_LABELS:
        return False

    sorted_scores = np.sort(predictions)[::-1]
    top_score = float(sorted_scores[0])
    second_score = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    confidence_margin = top_score - second_score

    return (
        top_score < constants.MIN_PEST_CONFIDENCE
        and confidence_margin < constants.MIN_CONFIDENCE_MARGIN
        and _non_pest_index(class_labels) is not None
    )

def predict_image(img_path, model=None, meta=None):
    """
    Predict pest from image
    Returns: (class_index, confidence_score)
    """
    if model is None or meta is None:
        model, meta = load_artifacts()
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    prediction_scores = predictions[0]
    predicted_class = int(np.argmax(prediction_scores))
    confidence = float(np.max(prediction_scores))

    class_labels = load_class_labels()
    if _should_reject_prediction(prediction_scores, predicted_class, class_labels):
        predicted_class = _non_pest_index(class_labels)
    
    return predicted_class, confidence

if __name__ == "__main__":
    # Test the function
    test_image = os.path.join(SCRIPT_DIR, "test.jpg")
    if os.path.exists(test_image):
        class_idx, conf = predict_image(test_image)
        print(f"Predicted class: {class_idx}, Confidence: {conf:.2f}")