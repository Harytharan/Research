import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

def evaluate_and_report(model, X_test, y_test, label_encoder, out_json=None):
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        pass

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": label_encoder.classes_.tolist()
    }

    print("Accuracy:", acc)
    print("Classification Report (text):")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {out_json}")

    return metrics
