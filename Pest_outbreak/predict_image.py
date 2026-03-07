# Pest_outbreak/predict_image.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use absolute paths
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_pest_model.h5")
META_PATH = os.path.join(SCRIPT_DIR, "paddy_meta.joblib")

def load_artifacts():
    """Load model and metadata"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata not found at: {META_PATH}")
    
    model = load_model(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta

def predict_image(img_path, model=None, meta=None):
    """
    Predict pest from image
    Returns: (class_index, confidence_score)
    """
    if model is None or meta is None:
        model, meta = load_artifacts()
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    return predicted_class, confidence

if __name__ == "__main__":
    # Test the function
    test_image = os.path.join(SCRIPT_DIR, "test.jpg")
    if os.path.exists(test_image):
        class_idx, conf = predict_image(test_image)
        print(f"Predicted class: {class_idx}, Confidence: {conf:.2f}")