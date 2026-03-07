import os
import numpy as np
import joblib
import predict_image as prt
import visualize_prediction as Visual

# -------------------------------------------------
# Base Directory
 
# -------------------------------------------------
# Paths
# -------------------------------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_pest_model.h5")
META_PATH = os.path.join(BASE_DIR, "paddy_meta.joblib")
TRAIN_PATH = os.path.join(BASE_DIR,  "PestDataset", "train")

# ✅ Correct way for image inside ImageBase folder
TEST_IMAGE_PATH = os.path.join(BASE_DIR,  "1.jpg")

# -------------------------------------------------
# Load Metadata
# -------------------------------------------------
print("[INFO] Loading metadata...")
if not os.path.exists(META_PATH):
    print("❌ Metadata file not found:", META_PATH)
    exit()

meta = joblib.load(META_PATH)

# Convert action_map keys to lowercase (SAFE FIX)
raw_action_map = meta.get("action_map", {})
action_map = {k.lower(): v for k, v in raw_action_map.items()}

# -------------------------------------------------
# Get Class Labels
# -------------------------------------------------
def get_class_labels(train_path):
    if not os.path.exists(train_path):
        print("❌ Train folder not found:", train_path)
        exit()

    class_names = sorted([
        d for d in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, d))
    ])
    return class_names

class_labels = get_class_labels(TRAIN_PATH)

# -------------------------------------------------
# Main Function
# -------------------------------------------------
def main():

    if not os.path.exists(TEST_IMAGE_PATH):
        print("❌ Test image not found:", TEST_IMAGE_PATH)
        return

    print("[INFO] Running prediction...")

    # Get prediction from CNN
    predicted_class_idx, confidence_score = prt.predict_image(TEST_IMAGE_PATH)

    predicted_class_name = class_labels[predicted_class_idx]

    # Get recommended action (CASE INSENSITIVE FIX)
    recommended_action = action_map.get(
        predicted_class_name.lower(),
        "No recommended action found."
    )

    # Visualize prediction
    Visual.visualize_prediction(
        TEST_IMAGE_PATH,
        predicted_class_name,
        confidence_score
    )

    # -------------------------------------------------
    # Final Output
    # -------------------------------------------------
    print("\n===== FINAL RESULT =====")
    print(f"Predicted Pest      : {predicted_class_name}")
    print(f"Confidence Score    : {confidence_score:.2f}")
    print(f"Recommended Action  : {recommended_action}")
    print("========================\n")


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    main()