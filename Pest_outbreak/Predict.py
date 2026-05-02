import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import visualize_prediction as Visual
import predict_image as prt

# Define constants
IMG_SIZE = (224, 224)
MODEL_PATH = "best_pest_model.h5"
TEST_IMAGE_PATH = "1.jpg"

# Load the trained model
model = load_model(MODEL_PATH)
class_labels = prt.load_class_labels()


# prediction
def main():
    img_path = TEST_IMAGE_PATH

    if not os.path.exists(img_path):
        print("The specified image path does not exist.")
        return

    # Get predicted class
    predicted_class_idx, confidence_score = prt.predict_image(img_path)
    if predicted_class_idx >= len(class_labels):
        print("Predicted class index is out of range for the saved class labels.")
        return

    predicted_class_name = class_labels[predicted_class_idx]
    display_name = "Non-pest item" if predicted_class_name.lower() in prt.NON_PEST_LABELS else predicted_class_name

    # Visualize
    Visual.visualize_prediction(img_path, display_name, confidence_score)

    print(f"Predicted Pest Type: {display_name}")
    print(f"Prediction Confidence: {confidence_score:.2f}")


if __name__ == "__main__":
    main()
