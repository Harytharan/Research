import serial
import time
import re
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings("ignore")


MODEL_PATH = "paddy_pest_rf.joblib"
META_PATH = "paddy_meta.joblib"

#  LOAD MODEL
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    print("[OK] ML model + metadata loaded")
    return model, meta


# PREDICT
def predict(values, model, meta):
    arr = np.array(values, dtype=float).reshape(1, -1)
    scaler = meta["scaler"]
    X_scaled = scaler.transform(arr)

    pred_idx = model.predict(X_scaled)[0]
    label = meta["label_encoder"].inverse_transform([pred_idx])[0]

    try:
        prob = float(np.max(model.predict_proba(X_scaled)[0]))
    except:
        prob = None

    action_map = meta.get("action_map", {})
    action = action_map.get(label, "No recommended action found.")

    return {
        "predicted_pest": label,
        "probability": prob,
        "recommended_action": action
    }

#  READ
def read_one_group(port="COM3", baudrate=115200):
    ser = serial.Serial(port, baudrate, timeout=1)
    print("[INFO] Waiting for ONE complete sensor cycle...\n")

    temp = None
    hum = 0
    press = None
    light = None

    while True:
        raw = ser.readline()
        if not raw:
            continue

        try:
            text = raw.decode().strip()
        except:
            text = raw.decode(errors="ignore").strip()

        print("[RAW]", text)


        if "Temperature" in text:
            m = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            if m:
                temp = float(m[0])


        elif "Pressure" in text:
            m = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            if m:
                press = float(m[0])


        elif "LDR Analog Value" in text:
            m = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            if m:
                light = float(m[0])


        if "------------------------" in text:
            if None not in (temp, press, light):
                ser.close()
                print("\n[OK] Complete packet received!")
                return [temp, hum, press, light]
            else:
                print("[WARN] Packet incomplete, waiting for next set...")


if __name__ == "__main__":
    model, meta = load_artifacts()

    values = read_one_group()
    print("\nSENSOR VALUES:", values)

    result = predict(values, model, meta)

    print("\nFINAL ML PREDICTION:")
    print(json.dumps(result, indent=2))
