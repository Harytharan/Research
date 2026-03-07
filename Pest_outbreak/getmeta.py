import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your CSV dataset
df = pd.read_csv("paddy_pest_dataset.csv")

# Select features
feature_columns = ["Temperature (Â°C)", "Humidity (%)", "Pressure (hPa)", "Light Intensity (lux)"]

# Encode the target pest labels
le_pest = LabelEncoder()
df["Detected Pest Encoded"] = le_pest.fit_transform(df["Detected Pest"])

# Optional: Encode paddy stage if you plan to use it
le_stage = LabelEncoder()
df["Paddy Stage Encoded"] = le_stage.fit_transform(df["Paddy Stage"])

# Create action map: Pest → Recommended Action
action_map = df.set_index("Detected Pest")["Recommended Action"].to_dict()

# Prepare metadata dictionary
meta = {
    "label_encoder": le_pest,
    "stage_encoder": le_stage,
    "feature_columns": feature_columns,
    "action_map": action_map
}

# Save metadata to joblib
joblib.dump(meta, "ImageBase/paddy_meta.joblib")

print("✅ paddy_meta.joblib created successfully!")