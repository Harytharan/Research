import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "paddy_disease_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")
PEST_SOLUTIONS_PATH = os.path.join(BASE_DIR, "pest_solutions.json")
NON_LEAF_LABELS = {"none", "not_leaf", "non_leaf"}

# Load model and class indices
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Inverse mapping
class_indices = {v: k for k, v in class_indices.items()}

# Load pest solutions
with open(PEST_SOLUTIONS_PATH, "r") as f:
    pest_data = json.load(f)


def predict_paddy_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    disease = class_indices[class_idx]

    if disease.lower() in NON_LEAF_LABELS:
        return {
            "disease": "not_leaf",
            "pest": "Not a paddy leaf",
            "solution": "The uploaded image does not appear to be a paddy leaf. Please upload a clear leaf image.",
            "confidence": confidence,
            "is_leaf": False
        }

    details = pest_data.get(disease, {})
    return {
        "disease": disease,
        "pest": details.get("pest", "Unknown pest"),
        "solution": details.get("solution", "No solution available."),
        "confidence": confidence,
        "is_leaf": True
    }