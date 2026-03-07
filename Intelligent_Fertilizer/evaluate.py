import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

DATA_FILE = "dataset.csv"

def evaluate():

    # Load dataset
    df = pd.read_csv(DATA_FILE)


    # Load saved models
    clf = joblib.load("models/fertilizer_model.pkl")
    reg = joblib.load("models/yield_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    feature_cols = joblib.load("models/feature_names.pkl")


    # Encode categorical columns
    for col, le in encoders.items():
        df[col] = le.transform(df[col])


    # Prepare targets
    X = df[feature_cols]
    y_fertilizer = df["Recommended_Fertilizer"]
    y_yield = df["Predicted_Yield_ton_per_ha"]


    # Scale features
    X_scaled = scaler.transform(X)


    # Train andtest split
    X_train, X_test, yF_train, yF_test, yY_train, yY_test = train_test_split(
        X_scaled,
        y_fertilizer,
        y_yield,
        test_size=0.2,
        random_state=42
    )


    # Predictions

    yF_pred = clf.predict(X_test)
    yY_pred = reg.predict(X_test)


    # Metrics
    acc = accuracy_score(yF_test, yF_pred)
    mse = mean_squared_error(yY_test, yY_pred)
    r2 = r2_score(yY_test, yY_pred)



    print("\n--> MODEL EVALUATION ")
    print(f"Fertilizer Classification Accuracy : {acc * 100:.2f}%")
    print(f"Yield Prediction MSE               : {mse:.3f}")
    print(f"Yield Prediction R² Score          : {r2:.3f}")

    print("\n--> Evaluation completed successfully")


if __name__ == "__main__":
    evaluate()
