import serial
import time
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from load import load_dataset
from pre_process import preprocess
from predict import (
    add_rolling_and_seasonal,
    add_price_momentum,
    predict_price_future_enhanced,
    predict_demand_future_enhanced
)

 
PORT = "COM3"
BAUD = 115200
N_DAYS = 7
 

def parse_npk(line):
    try:
        parts = line.replace(" ", "").split(",")
        n = float(parts[0].split(":")[1])
        p = float(parts[1].split(":")[1])
        k = float(parts[2].split(":")[1])
        return n, p, k
    except:
        return None

def main():
    print("🌱 Live NPK Sensor Prediction System")

    # Load models
    lstm = load_model("models/lstm_price_model_final.h5")
    xgb = joblib.load("models/xgb_demand_model_best_optimized.joblib")
    feature_cols_xgb = joblib.load("models/feature_columns_optimized.joblib")
    feature_cols_lstm = joblib.load("models/lstm_feature_columns.joblib")
    training_info = joblib.load("models/training_info.joblib")
    window_size = training_info["window_size"]

    # Load dataset
    df = load_dataset()
    df_raw, df_mm, df_std, _ = preprocess(df, save_artifacts=False)

    # Feature engineering
    df_mm = add_rolling_and_seasonal(df_mm)
    df_mm = add_price_momentum(df_mm)
    df_mm = df_mm.dropna().reset_index(drop=True)

    df_std = add_rolling_and_seasonal(df_std)
    df_std = add_price_momentum(df_std)
    df_std = df_std.dropna().reset_index(drop=True)

    # Serial connection
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    print("--> Waiting for sensor data...")

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        parsed = parse_npk(line)

        if parsed:
            n, p, k = parsed
            print(f"--> Sensor → N:{n} P:{p} K:{k}")

            # Inject sensor values into latest row
            df_mm.loc[df_mm.index[-1], "Nitrogen_N"] = n
            df_mm.loc[df_mm.index[-1], "Phosphorus_P"] = p
            df_mm.loc[df_mm.index[-1], "Potassium_K"] = k

            # Predict price
            start_date = pd.Timestamp.now() + pd.Timedelta(days=1)
            price_preds, dates = predict_price_future_enhanced(
                df_mm, lstm, feature_cols_lstm, start_date,
                n_steps=N_DAYS, window_size=window_size
            )

            # Predict demand
            demand_preds = predict_demand_future_enhanced(
                df_std, xgb, feature_cols_xgb, price_preds, dates
            )

            print("\n--> PREDICTIONS")
            for d, pr, dm in zip(dates, price_preds, demand_preds):
                print(f"{d.date()} → Price: {pr:.2f} LKR/kg | Demand: {dm:.2f} Tons")

            print("-" * 50)
            time.sleep(5)

if __name__ == "__main__":
    main()
