import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# Load model and class indices
model = tf.keras.models.load_model("paddy_disease_model.h5", compile=False)
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Inverse mapping
class_indices = {v: k for k, v in class_indices.items()}

# Load pest solutions
with open("pest_solutions.json", "r") as f:
    pest_data = json.load(f)

def predict_paddy_disease(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    disease = class_indices[class_idx]
    
    pest = pest_data[disease]["pest"]
    solution = pest_data[disease]["solution"]
    
    return {
        "disease": disease,
        "pest": pest,
        "solution": solution
    }

# Example usage
result = predict_paddy_disease("t.jpg")
print(result)