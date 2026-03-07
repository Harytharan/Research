import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42

def load_csv(path="paddy_pest_dataset.csv"):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def feature_engineering(df):
    df2 = df.copy()

    def get_col(cols, choices):
        for c in choices:
            if c in cols:
                return c
        return None

    cols = df2.columns.tolist()

    temp_col = get_col(cols, ["Temperature (Â°C)", "Temperature (°C)", "Temperature", "Air_Temperature_C"])
    hum_col  = get_col(cols, ["Humidity (%)", "Humidity"])
    pres_col = get_col(cols, ["Pressure (hPa)", "Pressure"])
    light_col= get_col(cols, ["Light Intensity (lux)", "Light_Intensity_lux"])
    stage_col= get_col(cols, ["Paddy Stage", "Growth_Stage"])
    pest_col = get_col(cols, ["Detected Pest", "Pest_Name", "Pest"])
    action_col = get_col(cols, ["Recommended Action", "Action", "Pest_Action"])

    # Required columns must exist
    if None in (temp_col, hum_col, pres_col, light_col, stage_col, pest_col):
        missing = [name for name, col in [
            ("Temperature", temp_col), ("Humidity", hum_col), ("Pressure", pres_col),
            ("Light", light_col), ("Stage", stage_col), ("Pest", pest_col)
        ] if col is None]
        raise ValueError(f"Missing required columns: {missing}")

    # Convert numeric inputs
    for c in [temp_col, hum_col, pres_col, light_col]:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    # Optional: Day of Year
    df2["DayOfYear"] = 0  # your dataset has no Date

    # Features
    X = df2[[temp_col, hum_col, pres_col, light_col]].copy()
    X.columns = ["temperature", "humidity", "pressure", "light"]

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df2[pest_col])

    # Build action map from dataset
    action_map = {}
    if action_col is not None:
        for pest, act in zip(df2[pest_col], df2[action_col]):
            action_map[str(pest).strip()] = str(act).strip()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    meta = {
        "label_encoder": le,
        "scaler": scaler,
        "feature_columns": ["temperature", "humidity", "pressure", "light"],
        "action_map": action_map
    }

    return X_train, X_test, y_train, y_test, meta


def preprocess_from_file(path="paddy_pest_dataset.csv"):
    df = load_csv(path)
    return feature_engineering(df)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, meta = preprocess_from_file()
    print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("Label classes:", meta["label_encoder"].classes_)
