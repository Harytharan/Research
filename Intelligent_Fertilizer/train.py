import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

DATA_FILE = "Dataset.csv"

def main():
    df = pd.read_csv(DATA_FILE)

    label_encoders = {}
    categorical_cols = [
        "Paddy_Growth_Stage",
        "Purpose",
        "Recommended_Fertilizer"
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le


    # FEATURES
    feature_cols = [
        "Soil_Temperature (°C)",
        "Soil_Moisture (%)",
        "Air_Temperature (°C)",
        "Air_Humidity (%)",
        "Paddy_Growth_Stage",
        "Purpose",
        "Recommended_Fertilizer",
        "Quantity_kg_per_acre"
    ]

    X = df[feature_cols]

    # Targets
    y_fertilizer = df["Recommended_Fertilizer"]
    y_yield = df["Predicted_Yield_ton_per_ha"]


    # Scaling

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, yF_train, yF_test, yY_train, yY_test = train_test_split(
        X_scaled, y_fertilizer, y_yield, test_size=0.2, random_state=42
    )


    # Models

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    reg = RandomForestRegressor(n_estimators=300, random_state=42)

    clf.fit(X_train, yF_train)
    reg.fit(X_train, yY_train)


    # Evaluation
    acc = accuracy_score(yF_test, clf.predict(X_test))
    mse = mean_squared_error(yY_test, reg.predict(X_test))
    r2 = r2_score(yY_test, reg.predict(X_test))

    print("\n--> MODEL PERFORMANCE")
    print(f"Classification Accuracy for Fertilizer: {acc*100:.2f}%")
    print(f"Regression MSE for Predicted Yield: {mse:.3f}")
    print(f"Regression R2 score: {r2:.3f}")


    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "fertilizer_models/fertilizer_model.pkl")
    joblib.dump(reg, "yield_models/yield_model.pkl")
    joblib.dump(scaler, "yield_models/scaler.pkl")
    joblib.dump(label_encoders, "fertilizer_models/label_encoders.pkl")
    joblib.dump(feature_cols, "fertilizer_models/feature_names.pkl")

    print("\n--> Training complete. Models saved")

if __name__ == "__main__":
    main()
