import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from load import load_dataset
from pre_process import preprocess_data
from featureeng import create_features

def train_model():
    df = load_dataset()
    df, encoders = preprocess_data(df)
    X, y = create_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=15
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Training complete!")

    # Evaluation
    print("\nModel Evaluation:")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

    # Save model and encoders
    joblib.dump(model, "trained_paddy_cost_model.pkl")
    joblib.dump(encoders, "label_encoders.pkl")
    print("\nModel and encoders saved successfully!")

if __name__ == "__main__":
    train_model()
