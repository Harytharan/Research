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
TRAIN_PATH = "PestDataset/train"

# Load the trained model
model = load_model(MODEL_PATH)

#  class labels
def get_class_labels(train_path):
    class_names = sorted(os.listdir(train_path))
    return class_names

class_labels = get_class_labels(TRAIN_PATH)


# prediction
def main():
    img_path = TEST_IMAGE_PATH

    if not os.path.exists(img_path):
        print("The specified image path does not exist.")
        return

    # Get predicted class
    predicted_class_idx, confidence_score = prt.predict_image(img_path)

    predicted_class_name = class_labels[predicted_class_idx]

    # Visualize
    Visual.visualize_prediction(img_path, predicted_class_name, confidence_score)

    print(f"Predicted Class: {predicted_class_name}")
    print(f"Prediction Confidence: {confidence_score:.2f}")


if __name__ == "__main__":
    main()
