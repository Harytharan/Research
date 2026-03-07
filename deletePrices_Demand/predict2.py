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
    predict_price_with_trend,
    predict_demand_with_trend
)


PORT = "COM3"
BAUD = 115200
N_DAYS = 7


def parse_npk(line):
    """Parse NPK sensor data"""
    try:
        # Clean the line
        line = line.strip()
        # Look for NPK pattern
        if "N:" in line and "P:" in line and "K:" in line:
            # Remove spaces and split
            parts = line.replace(" ", "").split(",")
            n = float(parts[0].split(":")[1])
            p = float(parts[1].split(":")[1])
            k = float(parts[2].split(":")[1])
            return n, p, k
    except Exception as e:
        print(f"Parse error: {e}")
    return None


def main():
    print("=" * 60)
    print("🌱 LIVE SENSOR-BASED PREDICTION SYSTEM")
    print("=" * 60)
    
    # Load models
    print("\n📂 Loading models...")
    try:
        lstm = load_model("models/lstm_price_model_final.h5")
        print("✅ LSTM price model loaded")
        
        xgb = joblib.load("models/xgb_demand_model_best_optimized.joblib")
        print("✅ XGBoost demand model loaded")
        
        feature_cols_xgb = joblib.load("models/feature_columns_optimized.joblib")
        feature_cols_lstm = joblib.load("models/lstm_feature_columns.joblib")
        training_info = joblib.load("models/training_info.joblib")
        window_size = training_info["window_size"]
        
        print(f"✅ Models loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Load and prepare dataset
    print("\n📊 Loading historical data...")
    df = load_dataset()
    df_raw, df_mm, df_std, _ = preprocess(df, save_artifacts=False)
    
    # Apply feature engineering
    print("⚙️ Preparing data...")
    df_mm = add_rolling_and_seasonal(df_mm)
    df_mm = add_price_momentum(df_mm)
    df_mm = df_mm.dropna().reset_index(drop=True)
    
    # Serial connection
    print(f"\n🔌 Connecting to sensor on {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)
        print("✅ Connected to sensor")
    except Exception as e:
        print(f"❌ Could not connect to sensor: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🔄 Waiting for sensor data...")
    print("=" * 60)
    
    last_prediction_time = 0
    
    while True:
        try:
            # Read from sensor
            line = ser.readline().decode(errors="ignore").strip()
            
            if line:
                parsed = parse_npk(line)
                
                if parsed:
                    n, p, k = parsed
                    current_time = time.time()
                    
                    print(f"\n📟 Sensor Reading @ {time.strftime('%H:%M:%S')}")
                    print(f"   Nitrogen (N): {n:.1f} mg/kg")
                    print(f"   Phosphorus (P): {p:.1f} mg/kg")
                    print(f"   Potassium (K): {k:.1f} mg/kg")
                    
                    # Update the latest row with sensor values
                    df_mm.loc[df_mm.index[-1], "Nitrogen_N"] = n
                    df_mm.loc[df_mm.index[-1], "Phosphorus_P"] = p
                    df_mm.loc[df_mm.index[-1], "Potassium_K"] = k
                    
                    # Make predictions every 30 seconds or on first run
                    if current_time - last_prediction_time > 30 or last_prediction_time == 0:
                        
                        # Predict price
                        start_date = pd.Timestamp.now() + pd.Timedelta(days=1)
                        price_preds, dates = predict_price_with_trend(
                            df_mm, lstm, feature_cols_lstm, start_date,
                            n_steps=N_DAYS, window_size=window_size
                        )
                        
                        # Predict demand
                        demand_preds = predict_demand_with_trend(
                            df_std, xgb, feature_cols_xgb, price_preds, dates
                        )
                        
                        # Display predictions
                        print("\n" + "=" * 60)
                        print("🔮 PREDICTIONS BASED ON SENSOR DATA")
                        print("=" * 60)
                        
                        for i, (d, pr, dm) in enumerate(zip(dates, price_preds, demand_preds)):
                            day_str = d.strftime('%Y-%m-%d')
                            print(f"\n📅 {day_str}")
                            print(f"   💰 Price: {pr:.2f} LKR/kg")
                            print(f"   📦 Demand: {dm:.2f} Tons")
                            
                            # Show day-over-day change
                            if i > 0:
                                price_change = pr - price_preds[i-1]
                                demand_change = dm - demand_preds[i-1]
                                print(f"   📊 Change: Price {price_change:+.2f}, Demand {demand_change:+.2f}")
                        
                        last_prediction_time = current_time
                        
                        print("\n" + "=" * 60)
                        print("⏳ Waiting for next sensor reading...")
                        print("=" * 60)
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n\n👋 Stopping sensor monitoring...")
            ser.close()
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()