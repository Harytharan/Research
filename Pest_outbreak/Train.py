import joblib
import os
from load_preprocess import preprocess_from_file
from fineTune import fine_tune_random_forest
from evaluate_model import evaluate_and_report
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "paddy_pest_dataset.csv"
MODEL_PATH = "paddy_pest_rf.joblib"
META_PATH = "paddy_meta.joblib"
METRICS_JSON = "paddy_metrics.json"

def main():
    print("Loading & preprocessing data...")
    X_train, X_test, y_train, y_test, meta = preprocess_from_file(DATA_PATH)

    print("Attempting fine-tuning with GridSearch...")
    try:
        best_model = fine_tune_random_forest(X_train, y_train)
    except Exception as e:
        print("Grid search failed or took too long—falling back to default RandomForest.")
        print("Error:", e)
        best_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        best_model.fit(X_train, y_train)

    # ensure model is fitted:
    try:
        # check if fitted
        _ = best_model.predict(X_test[:1])
    except Exception:
        best_model.fit(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_and_report(best_model, X_test, y_test, meta["label_encoder"], out_json=METRICS_JSON)

    # Save model  
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(meta, META_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Meta saved to {META_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()
