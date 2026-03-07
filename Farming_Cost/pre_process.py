
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    df = df.dropna()

    # Encode categorical features
    label_encoders = {}
    categorical_cols = ["Season", "Soil_Type", "Seed_Type"]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    print("Preprocessing complete. Encoded columns:", categorical_cols)
    return df, label_encoders

if __name__ == "__main__":
    from load import load_dataset
    df = load_dataset()
    df, encoders = preprocess_data(df)
    print(df.head())
