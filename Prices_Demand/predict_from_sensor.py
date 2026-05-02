# predict_from_sensor.py
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

from load import load_dataset
from pre_process import preprocess
from utils import add_rolling_and_seasonal, add_price_momentum
from run_prediction import predict_price_with_trend, predict_demand_with_trend


def get_user_inputs():
    print("\n" + "=" * 60)
    print("🌱 SENSOR-BASED PREDICTION INPUT")
    print("=" * 60)

    print("\n📅 PREDICTION DATE")
    print("-" * 30)
    while True:
        try:
            date_str = input("Enter prediction start date (YYYY-MM-DD) or press Enter for tomorrow: ").strip()
            if date_str == "":
                start_date = pd.Timestamp.now() + pd.Timedelta(days=1)
            else:
                start_date = pd.to_datetime(date_str)

            n_days = input("Enter number of days to predict (default 7): ").strip()
            n_days = 7 if n_days == "" else int(n_days)

            if n_days > 0:
                break
            else:
                print("Please enter a positive number of days")
        except ValueError:
            print("Invalid input. Please try again.")

    print("\n🗺️ REGION")
    print("-" * 30)
    print("Available regions: North, South, East, West, Central, North Central, Uva, Sabaragamuwa")
    region = input("Enter region: ").strip()
    if region == "":
        region = "North"

    print("\n🌾 RICE TYPE")
    print("-" * 30)
    print("Available rice types: Nadu, Samba, Keeri Samba")
    rice_type = input("Enter rice type: ").strip()
    if rice_type == "":
        rice_type = "Samba"

    print("\n🌱 SENSOR READINGS (NPK VALUES)")
    print("-" * 30)
    while True:
        try:
            n = float(input("   Nitrogen (N): "))
            p = float(input("   Phosphorus (P): "))
            k = float(input("   Potassium (K): "))

            confirm = input("\nConfirm NPK values? (y/n): ").strip().lower()
            if confirm in ["y", "yes", ""]:
                break
        except ValueError:
            print("Invalid input. Please enter numeric values.")

    return {
        "start_date": start_date,
        "n_days": n_days,
        "region": region,
        "rice_type": rice_type,
        "nitrogen": n,
        "phosphorus": p,
        "potassium": k
    }


def main():
    print("=" * 60)
    print("🌱 SINGLE-RUN SENSOR-BASED PREDICTION SYSTEM")
    print("=" * 60)

    user_inputs = get_user_inputs()

    print("\n📂 Loading models...")
    try:
        lstm = load_model("models/lstm_price_model_final.h5")
        print("✅ LSTM price model loaded")

        xgb = joblib.load("models/xgb_demand_model.joblib")
        print("✅ XGBoost demand model loaded")

        feature_cols_xgb = joblib.load("models/xgb_feature_columns.joblib")
        feature_cols_lstm = joblib.load("models/lstm_feature_columns.joblib")
        training_info = joblib.load("models/training_info.joblib")

        region_encoder = joblib.load("models/region_encoder.joblib")
        rice_type_encoder = joblib.load("models/rice_type_encoder.joblib")

        window_size = training_info["window_size"]

        print("✅ All models and encoders loaded successfully")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nPlease ensure models are trained first (run train.py)")
        return

    print("\n📊 Loading historical data...")
    df = load_dataset()

    # preprocess
    df_raw, df_mm, df_std, _ = preprocess(df, save_artifacts=False)

    # make sure dates are datetime
    df_mm["Date"] = pd.to_datetime(df_mm["Date"])
    df_std["Date"] = pd.to_datetime(df_std["Date"])

    print("⚙️ Preparing data with sensor values...")
    df_mm = add_rolling_and_seasonal(df_mm)
    df_mm = add_price_momentum(df_mm)
    df_mm = df_mm.dropna().reset_index(drop=True)

    # keep only selected Region + Rice_Type group FIRST
    print("\n🔧 Applying sensor readings to selected historical group...")

    df_mm = df_mm[
        (df_mm["Region"] == user_inputs["region"]) &
        (df_mm["Rice_Type"] == user_inputs["rice_type"])
    ].copy().sort_values("Date").reset_index(drop=True)

    df_std = df_std[
        (df_std["Region"] == user_inputs["region"]) &
        (df_std["Rice_Type"] == user_inputs["rice_type"])
    ].copy().sort_values("Date").reset_index(drop=True)

    print(f"Selected group rows for LSTM: {len(df_mm)}")
    print(f"Selected group rows for XGBoost: {len(df_std)}")

    if len(df_mm) < window_size:
        print(
            f"❌ Not enough historical rows for {user_inputs['region']} + {user_inputs['rice_type']} "
            f"for LSTM window size {window_size}. Found only {len(df_mm)} rows."
        )
        return

    try:
        # apply new NPK values to last row of selected group only
        df_mm.loc[df_mm.index[-1], "Nitrogen_N"] = user_inputs["nitrogen"]
        df_mm.loc[df_mm.index[-1], "Phosphorus_P"] = user_inputs["phosphorus"]
        df_mm.loc[df_mm.index[-1], "Potassium_K"] = user_inputs["potassium"]

        df_std.loc[df_std.index[-1], "Nitrogen_N"] = user_inputs["nitrogen"]
        df_std.loc[df_std.index[-1], "Phosphorus_P"] = user_inputs["phosphorus"]
        df_std.loc[df_std.index[-1], "Potassium_K"] = user_inputs["potassium"]

        # ensure encoded values exist
        df_mm.loc[df_mm.index[-1], "Region_encoded"] = region_encoder.transform([user_inputs["region"]])[0]
        df_mm.loc[df_mm.index[-1], "Rice_Type_encoded"] = rice_type_encoder.transform([user_inputs["rice_type"]])[0]

        df_std.loc[df_std.index[-1], "Region_encoded"] = region_encoder.transform([user_inputs["region"]])[0]
        df_std.loc[df_std.index[-1], "Rice_Type_encoded"] = rice_type_encoder.transform([user_inputs["rice_type"]])[0]

    except Exception as e:
        print(f"❌ Error encoding region or rice type: {e}")
        print("Make sure the region and rice type match the training dataset values exactly.")
        return

    print(f"   Region: {user_inputs['region']}")
    print(f"   Rice Type: {user_inputs['rice_type']}")
    print(f"   Nitrogen: {user_inputs['nitrogen']:.1f} mg/kg")
    print(f"   Phosphorus: {user_inputs['phosphorus']:.1f} mg/kg")
    print(f"   Potassium: {user_inputs['potassium']:.1f} mg/kg")

    # PRICE PREDICTION
    print(f"\n💰 Predicting prices for {user_inputs['n_days']} days...")
    try:
        price_predictions, prediction_dates = predict_price_with_trend(
            df_mm,
            lstm,
            feature_cols_lstm,
            user_inputs["start_date"],
            n_steps=user_inputs["n_days"],
            window_size=window_size
        )

        print("\n" + "=" * 50)
        print("PRICE PREDICTIONS")
        print("=" * 50)

        price_trend = np.polyfit(range(len(price_predictions)), price_predictions, 1)[0]

        for i, (date, price) in enumerate(zip(prediction_dates, price_predictions)):
            if i > 0:
                change = price - price_predictions[i - 1]
                change_pct = (change / price_predictions[i - 1]) * 100
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"   {date.strftime('%Y-%m-%d')}: {price:.2f} LKR/kg  {arrow} {change:+.2f} ({change_pct:+.1f}%)")
            else:
                print(f"   {date.strftime('%Y-%m-%d')}: {price:.2f} LKR/kg")

        print(f"\n   📈 Trend: {'Increasing' if price_trend > 0 else 'Decreasing' if price_trend < 0 else 'Stable'}")
        print(f"   📊 Range: {min(price_predictions):.2f} - {max(price_predictions):.2f} LKR/kg")

    except Exception as e:
        print(f"❌ Error in price prediction: {e}")
        import traceback
        traceback.print_exc()
        return

    # DEMAND PREDICTION
    print(f"\n📦 Predicting demand for {user_inputs['n_days']} days...")
    try:
        demand_predictions = predict_demand_with_trend(
            df_std,
            xgb,
            feature_cols_xgb,
            price_predictions,
            prediction_dates
        )

        print("\n" + "=" * 50)
        print("DEMAND PREDICTIONS")
        print("=" * 50)

        demand_trend = np.polyfit(range(len(demand_predictions)), demand_predictions, 1)[0]

        for i, (date, demand) in enumerate(zip(prediction_dates, demand_predictions)):
            if i > 0:
                change = demand - demand_predictions[i - 1]
                change_pct = (change / demand_predictions[i - 1]) * 100
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"   {date.strftime('%Y-%m-%d')}: {demand:.2f} Tons  {arrow} {change:+.2f} ({change_pct:+.1f}%)")
            else:
                print(f"   {date.strftime('%Y-%m-%d')}: {demand:.2f} Tons")

        print(f"\n   📈 Trend: {'Increasing' if demand_trend > 0 else 'Decreasing' if demand_trend < 0 else 'Stable'}")
        print(f"   📊 Range: {min(demand_predictions):.2f} - {max(demand_predictions):.2f} Tons")

    except Exception as e:
        print(f"❌ Error in demand prediction: {e}")
        import traceback
        traceback.print_exc()
        return

    # SUMMARY
    print("\n" + "=" * 50)
    print("📋 PREDICTION SUMMARY")
    print("=" * 50)
    print(
        f"Period: {user_inputs['start_date'].strftime('%Y-%m-%d')} to "
        f"{(user_inputs['start_date'] + timedelta(days=user_inputs['n_days'] - 1)).strftime('%Y-%m-%d')}"
    )

    print(f"\n📊 Price:")
    print(f"   Average: {np.mean(price_predictions):.2f} LKR/kg")
    print(f"   Min: {min(price_predictions):.2f} LKR/kg")
    print(f"   Max: {max(price_predictions):.2f} LKR/kg")

    print(f"\n📊 Demand:")
    print(f"   Average: {np.mean(demand_predictions):.2f} Tons")
    print(f"   Min: {min(demand_predictions):.2f} Tons")
    print(f"   Max: {max(demand_predictions):.2f} Tons")

    correlation = np.corrcoef(price_predictions, demand_predictions)[0, 1]
    print(f"\n📈 Price-Demand Correlation: {correlation:.3f}")

    results_df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in prediction_dates],
        "Region": user_inputs["region"],
        "Rice_Type": user_inputs["rice_type"],
        "Predicted_Price_LKR_per_kg": [f"{p:.2f}" for p in price_predictions],
        "Predicted_Demand_Tons": [f"{d:.2f}" for d in demand_predictions],
        "Nitrogen_mg_kg": user_inputs["nitrogen"],
        "Phosphorus_mg_kg": user_inputs["phosphorus"],
        "Potassium_mg_kg": user_inputs["potassium"]
    })

    filename = f"sensor_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")

    print("\n✅ Prediction completed successfully!")


if __name__ == "__main__":
    main()