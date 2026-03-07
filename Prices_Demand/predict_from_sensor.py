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
    """Get all inputs from user"""
    print("\n" + "=" * 60)
    print("🌱 SENSOR-BASED PREDICTION INPUT")
    print("=" * 60)
    
    # Date input
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
            if n_days == "":
                n_days = 7
            else:
                n_days = int(n_days)
            
            if n_days > 0:
                break
            else:
                print("Please enter a positive number of days")
        except ValueError:
            print("Invalid input. Please try again.")
    
    # NPK Sensor inputs
    print("\n🌱 SENSOR READINGS (NPK VALUES)")
    print("-" * 30)
    print("Enter the NPK sensor readings:")
    
    while True:
        try:
            n = float(input("   Nitrogen (N) [mg/kg, range 450-600]: "))
            p = float(input("   Phosphorus (P) [mg/kg, range 5-15]: "))
            k = float(input("   Potassium (K) [mg/kg, range 5-15]: "))
            
            # Optional validation with warnings
            if n < 400 or n > 650:
                print("⚠️ Warning: Nitrogen value outside typical range (450-600)")
            if p < 3 or p > 20:
                print("⚠️ Warning: Phosphorus value outside typical range (5-15)")
            if k < 3 or k > 20:
                print("⚠️ Warning: Potassium value outside typical range (5-15)")
            
            confirm = input("\nConfirm NPK values? (y/n): ").strip().lower()
            if confirm in ['y', 'yes', '']:
                break
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    
    # Optional: Weather inputs
    print("\n☁️ WEATHER CONDITIONS (Optional)")
    print("-" * 30)
    print("Press Enter to use default/average values")
    
    try:
        rainfall = input("   Rainfall (mm) [default: 50]: ").strip()
        rainfall = float(rainfall) if rainfall else 50.0
    except ValueError:
        rainfall = 50.0
        print("   Using default: 50.0 mm")
    
    try:
        temperature = input("   Temperature (°C) [default: 27]: ").strip()
        temperature = float(temperature) if temperature else 27.0
    except ValueError:
        temperature = 27.0
        print("   Using default: 27.0 °C")
    
    return {
        'start_date': start_date,
        'n_days': n_days,
        'nitrogen': n,
        'phosphorus': p,
        'potassium': k,
        'rainfall': rainfall,
        'temperature': temperature
    }


def main():
    print("=" * 60)
    print("🌱 SINGLE-RUN SENSOR-BASED PREDICTION SYSTEM")
    print("=" * 60)
    
    # Get user inputs
    user_inputs = get_user_inputs()
    
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
        
        print("✅ All models loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nPlease ensure models are trained first (run train.py)")
        return
    
    # Load and prepare dataset
    print("\n📊 Loading historical data...")
    df = load_dataset()
    df_raw, df_mm, df_std, _ = preprocess(df, save_artifacts=False)
    
    # Apply feature engineering
    print("⚙️ Preparing data with sensor values...")
    df_mm = add_rolling_and_seasonal(df_mm)
    df_mm = add_price_momentum(df_mm)
    df_mm = df_mm.dropna().reset_index(drop=True)
    
    # Update the latest row with user inputs
    print("\n🔧 Applying sensor readings to model...")
    df_mm.loc[df_mm.index[-1], "Nitrogen_N"] = user_inputs['nitrogen']
    df_mm.loc[df_mm.index[-1], "Phosphorus_P"] = user_inputs['phosphorus']
    df_mm.loc[df_mm.index[-1], "Potassium_K"] = user_inputs['potassium']
    
    # Update weather if provided
    if 'Rainfall_mm' in df_mm.columns:
        df_mm.loc[df_mm.index[-1], "Rainfall_mm"] = user_inputs['rainfall']
    if 'Temperature_C' in df_mm.columns:
        df_mm.loc[df_mm.index[-1], "Temperature_C"] = user_inputs['temperature']
    
    print(f"   Nitrogen: {user_inputs['nitrogen']:.1f} mg/kg")
    print(f"   Phosphorus: {user_inputs['phosphorus']:.1f} mg/kg")
    print(f"   Potassium: {user_inputs['potassium']:.1f} mg/kg")
    print(f"   Rainfall: {user_inputs['rainfall']:.1f} mm")
    print(f"   Temperature: {user_inputs['temperature']:.1f} °C")
    
    # ========== PRICE PREDICTION ==========
    print(f"\n💰 Predicting prices for {user_inputs['n_days']} days...")
    try:
        price_predictions, prediction_dates = predict_price_with_trend(
            df_mm, lstm, feature_cols_lstm, 
            user_inputs['start_date'], 
            n_steps=user_inputs['n_days'], 
            window_size=window_size
        )
        
        # Display price predictions
        print("\n" + "=" * 50)
        print("PRICE PREDICTIONS")
        print("=" * 50)
        
        price_trend = np.polyfit(range(len(price_predictions)), price_predictions, 1)[0]
        
        for i, (date, price) in enumerate(zip(prediction_dates, price_predictions)):
            if i > 0:
                change = price - price_predictions[i-1]
                change_pct = (change / price_predictions[i-1]) * 100
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
    
    # ========== DEMAND PREDICTION ==========
    print(f"\n📦 Predicting demand for {user_inputs['n_days']} days...")
    try:
        demand_predictions = predict_demand_with_trend(
            df_std, xgb, feature_cols_xgb, price_predictions, prediction_dates
        )
        
        # Display demand predictions
        print("\n" + "=" * 50)
        print("DEMAND PREDICTIONS")
        print("=" * 50)
        
        demand_trend = np.polyfit(range(len(demand_predictions)), demand_predictions, 1)[0]
        
        for i, (date, demand) in enumerate(zip(prediction_dates, demand_predictions)):
            if i > 0:
                change = demand - demand_predictions[i-1]
                change_pct = (change / demand_predictions[i-1]) * 100
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"   {date.strftime('%Y-%m-%d')}: {demand:.2f} Tons  {arrow} {change:+.2f} ({change_pct:+.1f}%)")
            else:
                print(f"   {date.strftime('%Y-%m-%d')}: {demand:.2f} Tons")
        
        print(f"\n   📈 Trend: {'Increasing' if demand_trend > 0 else 'Decreasing' if demand_trend < 0 else 'Stable'}")
        print(f"   📊 Range: {min(demand_predictions):.2f} - {max(demand_predictions):.2f} Tons")
        
    except Exception as e:
        print(f"❌ Error in demand prediction: {e}")
        return
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 50)
    print("📋 PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Period: {user_inputs['start_date'].strftime('%Y-%m-%d')} to "
          f"{(user_inputs['start_date'] + timedelta(days=user_inputs['n_days']-1)).strftime('%Y-%m-%d')}")
    print(f"\n📊 Price:")
    print(f"   Average: {np.mean(price_predictions):.2f} LKR/kg")
    print(f"   Min: {min(price_predictions):.2f} LKR/kg")
    print(f"   Max: {max(price_predictions):.2f} LKR/kg")
    print(f"\n📊 Demand:")
    print(f"   Average: {np.mean(demand_predictions):.2f} Tons")
    print(f"   Min: {min(demand_predictions):.2f} Tons")
    print(f"   Max: {max(demand_predictions):.2f} Tons")
    
    # Correlation
    correlation = np.corrcoef(price_predictions, demand_predictions)[0, 1]
    print(f"\n📈 Price-Demand Correlation: {correlation:.3f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in prediction_dates],
        'Predicted_Price_LKR_per_kg': [f"{p:.2f}" for p in price_predictions],
        'Predicted_Demand_Tons': [f"{d:.2f}" for d in demand_predictions],
        'Nitrogen_mg_kg': user_inputs['nitrogen'],
        'Phosphorus_mg_kg': user_inputs['phosphorus'],
        'Potassium_mg_kg': user_inputs['potassium'],
        'Rainfall_mm': user_inputs['rainfall'],
        'Temperature_C': user_inputs['temperature']
    })
    
    filename = f"sensor_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to: {filename}")
    
    print("\n✅ Prediction completed successfully!")


if __name__ == "__main__":
    main()