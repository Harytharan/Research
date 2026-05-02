from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile
import traceback

from predict_image import NON_PEST_LABELS, load_class_labels

app = Flask(__name__)

# ------------------- CONFIG -------------------
MODEL_PATH = "best_pest_model.h5"
RECOMMENDATION_FILE = "recommendations.json"
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize model, labels, and recommendations
model = None
class_labels = []
recommendations = {}

# ------------------- INITIALIZATION -------------------
def initialize_app():
    global model, class_labels, recommendations
    try:
        # Load model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
        model = load_model(MODEL_PATH)
        print("Pest detection model loaded successfully!")

        # Load class labels
        class_labels = load_class_labels()
        print(f"Class labels: {class_labels}")

        # Load recommendations
        if os.path.exists(RECOMMENDATION_FILE):
            with open(RECOMMENDATION_FILE, "r") as f:
                recommendations = json.load(f)
        else:
            recommendations = {
                "aphid": {
                    "Mild": "Use neem oil spray every 7 days. Apply potassium-rich fertilizer.",
                    "Moderate": "Apply insecticidal soap. Use imidacloprid-based systemic insecticide.",
                    "Severe": "Use pyrethrin-based spray. Remove heavily infected leaves. Apply NPK 10-10-10 fertilizer."
                },
                "default": {
                    "Mild": "Apply organic neem oil. Use balanced NPK fertilizer.",
                    "Moderate": "Use general insecticide. Apply potassium-rich fertilizer.",
                    "Severe": "Use systemic insecticide. Remove affected areas. Use complete NPK fertilizer."
                }
            }
        print("Recommendations loaded successfully!")
    except Exception as e:
        print(f"Error initializing app: {e}")
        traceback.print_exc()

# ------------------- PREDICTION -------------------
def predict_pest(img_path):
    """Predict pest from uploaded image"""
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        pest_name = class_labels[predicted_idx] if class_labels else f"class_{predicted_idx}"
        if pest_name.lower() in NON_PEST_LABELS:
            return "Non-pest item", confidence
        return pest_name, confidence
    except Exception as e:
        print(f"Error predicting pest: {e}")
        traceback.print_exc()
        return "Unknown", 0.0

# ------------------- LEAF ANALYSIS -------------------
def analyze_leaf(img_path):
    """Analyze leaf health"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_healthy = np.array([30, 40, 40])
        upper_healthy = np.array([90, 255, 255])
        healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy)
        affected_mask = cv2.bitwise_not(healthy_mask)

        kernel = np.ones((5,5), np.uint8)
        affected_mask = cv2.morphologyEx(affected_mask, cv2.MORPH_OPEN, kernel)
        affected_mask = cv2.morphologyEx(affected_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(affected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_pixels = img.shape[0]*img.shape[1]
        affected_pixels = cv2.countNonZero(affected_mask)
        affected_percent = (affected_pixels / total_pixels) * 100 if total_pixels>0 else 0

        if affected_percent < 10:
            severity = "Mild"
        elif affected_percent < 30:
            severity = "Moderate"
        else:
            severity = "Severe"

        return {
            "affected_percent": round(affected_percent, 2),
            "affected_spots": len(contours),
            "severity": severity
        }
    except Exception as e:
        print(f"Error analyzing leaf: {e}")
        traceback.print_exc()
        return None

# ------------------- RECOMMENDATION -------------------
def fertilizer_recommendation(pest, severity):
    try:
        if pest in recommendations:
            return recommendations[pest].get(severity, recommendations["default"][severity])
        return recommendations["default"][severity]
    except:
        return "Use general fertilizer."

# ------------------- API ENDPOINT -------------------
@app.route("/pest/analyze", methods=["POST"])
def analyze_pest():
    try:
        if "pest_image" not in request.files or "leaf_image" not in request.files:
            return jsonify({"error": "Both pest_image and leaf_image are required"}), 400

        # Save uploaded files
        pest_file = request.files["pest_image"]
        leaf_file = request.files["leaf_image"]
        pest_path = os.path.join(UPLOAD_FOLDER, pest_file.filename)
        leaf_path = os.path.join(UPLOAD_FOLDER, leaf_file.filename)
        pest_file.save(pest_path)
        leaf_file.save(leaf_path)

        # Run prediction and analysis
        pest, confidence = predict_pest(pest_path)
        leaf_analysis = analyze_leaf(leaf_path)
        fert_rec = fertilizer_recommendation(pest, leaf_analysis["severity"] if leaf_analysis else "Mild")

        # Remove uploaded files
        os.remove(pest_path)
        os.remove(leaf_path)

        result = {
            "status": "success",
            "pest_prediction": {
                "pest_name": pest,
                "confidence_percentage": f"{round(confidence*100,2)}%"
            },
            "leaf_analysis": leaf_analysis,
            "recommendation": {
                "fertilizer": fert_rec,
                "severity_level": leaf_analysis["severity"] if leaf_analysis else "Unknown"
            }
        }
        return jsonify(result)

    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/pest/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "class_labels_count": len(class_labels)
    })

if __name__ == "__main__":
    initialize_app()
    app.run(host="0.0.0.0", port=5001, debug=True)
