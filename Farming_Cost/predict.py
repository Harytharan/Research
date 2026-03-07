# Farming_Cost/predict.py
import joblib
import pandas as pd
import os

def predict_total_cost(input_data):
    """
    Predict total paddy farming cost for given input data (dict).
    Uses absolute paths to find model files.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build paths to model files
    model_path = os.path.join(script_dir, 'models', 'trained_paddy_cost_model.pkl')
    encoders_path = os.path.join(script_dir, 'models', 'label_encoders.pkl')
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Encoders file not found at: {encoders_path}")
    
    # Load model and encoders
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)

    df_input = pd.DataFrame([input_data])

    # Encode categorical columns using saved encoders
    for col, le in encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col])

    # Predict cost
    predicted_cost = model.predict(df_input)[0]
    return round(predicted_cost, 2)

if __name__ == "__main__":
    # Example GPS-based sample
    sample = {
        "Latitude": 7.35,
        "Longitude": 81.65,
        "Year": 2025,
        "Season": "Maha",
        "Soil_Type": "Clay",
        "Rainfall_mm": 2200,
        "Temperature_C": 30.5,
        "Humidity_%": 85,
        "Area_acres": 2.5,
        "Seed_Type": "BG352",
        "Seed_Cost (LKR)": 8000,
        "Fertilizer_Cost (LKR)": 15000,
        "Pesticide_Cost (LKR)": 4000,
        "Labor_Cost (LKR)": 25000,
        "Water_Cost (LKR)": 5000,
        "Machinery_Cost (LKR)": 10000,
        "Other_Costs (LKR)": 2000
    }

    cost = predict_total_cost(sample)
    print(f"Predicted Total Cost: LKR {cost}")