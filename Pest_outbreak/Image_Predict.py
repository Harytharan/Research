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

class_labels = prt.load_class_labels()

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
    if predicted_class_idx >= len(class_labels):
        print("❌ Predicted class index is out of range for the saved class labels.")
        return

    predicted_class_name = class_labels[predicted_class_idx]
    is_non_pest = predicted_class_name.lower() in prt.NON_PEST_LABELS
    display_name = "Non-pest item" if is_non_pest else predicted_class_name

    # Get recommended action (CASE INSENSITIVE FIX)
    if is_non_pest:
        recommended_action = "No pest detected in the image."
    else:
        recommended_action = action_map.get(
            predicted_class_name.lower(),
            "No recommended action found."
        )

    # Visualize prediction
    Visual.visualize_prediction(
        TEST_IMAGE_PATH,
        display_name,
        confidence_score
    )

    # -------------------------------------------------
    # Final Output
    # -------------------------------------------------
    print("\n===== FINAL RESULT =====")
    print(f"Predicted Pest Type : {display_name}")
    print(f"Confidence Score    : {confidence_score:.2f}")
    print(f"Recommended Action  : {recommended_action}")
    print("========================\n")


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    main()