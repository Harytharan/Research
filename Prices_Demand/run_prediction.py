# run_prediction.py
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from load import load_dataset
from pre_process import preprocess
from utils import (
    add_lag_features, add_rolling_and_seasonal, add_price_momentum,
    calculate_historical_trend, update_time_features
)


# ==================== TREND-BASED PRICE PREDICTION ====================

def predict_price_with_trend(df, lstm_model, feature_cols, start_date, n_steps=7, window_size=21):
    """Predict future prices while maintaining historical trends"""
    
    df_extended = df.copy()
    predictions = []
    
    # Calculate historical trends
    price_trend = calculate_historical_trend(df_extended, 'Paddy_Price_LKR_per_kg', days=60)
    
    # Get recent price history for trend reference
    recent_prices = df_extended['Paddy_Price_LKR_per_kg'].tail(30).values
    
    # Get the last window of data
    current_data = df_extended[feature_cols].values[-window_size:]
    
    for i in range(n_steps):
        # Prepare sequence for prediction
        sequence = current_data.reshape(1, window_size, len(feature_cols))
        
        # Get LSTM prediction
        lstm_pred = lstm_model.predict(sequence, verbose=0)[0][0]
        
        # Blend with trend-based prediction
        if len(recent_prices) > 0:
            # Calculate trend-based prediction
            if i == 0:
                # First prediction: use recent price + trend
                last_price = recent_prices[-1]
                trend_component = last_price * (1 + price_trend['trend_strength'])
            else:
                # Subsequent predictions: use previous prediction + trend
                last_price = predictions[-1]
                trend_component = last_price * (1 + price_trend['trend_strength'])
            
            # Apply seasonal adjustment
            trend_component *= price_trend['seasonal_factor']
            
            # Blend LSTM and trend (70% LSTM, 30% trend for stability)
            blended_price = 0.7 * lstm_pred + 0.3 * trend_component
        else:
            blended_price = lstm_pred
        
        # Ensure price doesn't deviate too far from historical range
        historical_mean = df_extended['Paddy_Price_LKR_per_kg'].tail(90).mean()
        historical_std = df_extended['Paddy_Price_LKR_per_kg'].tail(90).std()
        
        # Cap deviation to within 2 standard deviations of historical mean
        max_deviation = 2 * historical_std
        min_allowed = max(historical_mean - max_deviation, df_extended['Paddy_Price_LKR_per_kg'].min() * 0.8)
        max_allowed = min(historical_mean + max_deviation, df_extended['Paddy_Price_LKR_per_kg'].max() * 1.2)
        
        blended_price = np.clip(blended_price, min_allowed, max_allowed)
        predictions.append(blended_price)
        
        # Create new features for next prediction
        new_date = start_date + pd.Timedelta(days=i)
        new_features = create_future_features(df_extended, blended_price, new_date, feature_cols)
        current_data = np.vstack([current_data[1:], new_features])
        
        # Add to extended dataframe for next iteration
        new_row = df_extended.iloc[-1].copy()
        new_row['Paddy_Price_LKR_per_kg'] = blended_price
        new_row['Date'] = new_date
        new_row = update_time_features(new_row, new_date)
        
        df_extended = pd.concat([df_extended, pd.DataFrame([new_row])], ignore_index=True)
        
        # Update recent prices list
        recent_prices = np.append(recent_prices[1:], blended_price) if len(recent_prices) > 1 else np.array([blended_price])
    
    prediction_dates = [start_date + pd.Timedelta(days=i) for i in range(n_steps)]
    return predictions, prediction_dates


def create_future_features(df, predicted_price, date, feature_cols):
    """Create feature vector for future prediction"""
    
    # Start with the last row's features
    if len(df) > 0:
        new_features = df[feature_cols].iloc[-1].copy()
    else:
        new_features = np.zeros(len(feature_cols))
    
    # Update price if it's in features
    if 'Paddy_Price_LKR_per_kg' in feature_cols:
        price_idx = feature_cols.index('Paddy_Price_LKR_per_kg')
        new_features[price_idx] = predicted_price
    
    # Update time-based features
    time_updates = {
        'Month': date.month,
        'Quarter': date.quarter,
        'day_of_week': date.dayofweek,
        'day_of_year': date.dayofyear,
        'month_sin': np.sin(2 * np.pi * date.month / 12),
        'month_cos': np.cos(2 * np.pi * date.month / 12),
        'quarter_sin': np.sin(2 * np.pi * date.quarter / 4),
        'quarter_cos': np.cos(2 * np.pi * date.quarter / 4),
        'dow_sin': np.sin(2 * np.pi * date.dayofweek / 7),
        'dow_cos': np.cos(2 * np.pi * date.dayofweek / 7),
        'doy_sin': np.sin(2 * np.pi * date.dayofyear / 365),
        'doy_cos': np.cos(2 * np.pi * date.dayofyear / 365),
        'year_progress': date.dayofyear / 365.0,
        'is_weekend': 1 if date.dayofweek >= 5 else 0
    }
    
    for feature_name, value in time_updates.items():
        if feature_name in feature_cols:
            idx = feature_cols.index(feature_name)
            new_features[idx] = value
    
    return new_features.values if hasattr(new_features, 'values') else new_features


# ==================== TREND-BASED DEMAND PREDICTION ====================

def predict_demand_with_trend(df_original, xgb_model, feature_columns, price_predictions, prediction_dates):
    """Predict demand while maintaining historical trends"""
    
    # Create future dataframe
    future_df = create_future_dataframe(df_original, price_predictions, prediction_dates)
    
    # Calculate historical demand trend
    demand_trend = calculate_historical_trend(df_original, 'Demand_Tons', days=60)
    
    # Prepare features
    X_future = prepare_demand_features(future_df, feature_columns)
    
    # Get XGBoost predictions
    xgb_predictions = xgb_model.predict(X_future)
    
    # Blend with trend-based predictions
    trend_predictions = []
    recent_demand = df_original['Demand_Tons'].tail(30).values
    
    for i, (date, xgb_pred) in enumerate(zip(prediction_dates, xgb_predictions)):
        if len(recent_demand) > 0:
            if i == 0:
                last_demand = recent_demand[-1]
            else:
                last_demand = trend_predictions[-1]
            
            # Trend component
            trend_component = last_demand * (1 + demand_trend['trend_strength'])
            
            # Apply seasonal adjustment
            trend_component *= demand_trend['seasonal_factor']
            
            # Blend (60% XGBoost, 40% trend)
            blended_demand = 0.6 * xgb_pred + 0.4 * trend_component
        else:
            blended_demand = xgb_pred
        
        # Ensure demand stays within historical bounds
        historical_mean = df_original['Demand_Tons'].tail(90).mean()
        historical_std = df_original['Demand_Tons'].tail(90).std()
        
        max_deviation = 2 * historical_std
        min_allowed = max(historical_mean - max_deviation, df_original['Demand_Tons'].min() * 0.8)
        max_allowed = min(historical_mean + max_deviation, df_original['Demand_Tons'].max() * 1.2)
        
        blended_demand = np.clip(blended_demand, min_allowed, max_allowed)
        trend_predictions.append(blended_demand)
        
        # Update recent demand
        if len(recent_demand) > 0:
            recent_demand = np.append(recent_demand[1:], blended_demand)
    
    return np.array(trend_predictions)


def create_future_dataframe(df_original, price_predictions, prediction_dates):
    """Create dataframe for future predictions"""
    future_data = []
    
    # Get the last row as template
    last_row = df_original.iloc[-1].copy()
    
    for price, date in zip(price_predictions, prediction_dates):
        new_row = last_row.copy()
        new_row['Date'] = date
        new_row['Paddy_Price_LKR_per_kg'] = price
        new_row = update_time_features(new_row, date)
        future_data.append(new_row)
    
    future_df = pd.DataFrame(future_data)
    
    # Add engineered features
    future_df = add_lag_features(future_df, "Demand_Tons", n_lags=21)
    future_df = add_rolling_and_seasonal(future_df)
    future_df = add_price_momentum(future_df)
    
    return future_df


def prepare_demand_features(df_future, feature_columns):
    """Prepare features for demand prediction"""
    df_filled = df_future.copy()
    
    # Fill NaN values
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(0)
    
    # Ensure all required columns exist
    for col in feature_columns:
        if col not in df_filled.columns:
            df_filled[col] = 0
    
    return df_filled[feature_columns]


# ==================== VISUALIZATION FUNCTIONS ====================

def plot_predictions_with_history(historical_dates, historical_prices, historical_demand,
                                   prediction_dates, price_predictions, demand_predictions,
                                   start_date, end_date):
    """Plot predictions alongside historical data"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Convert to lists
    hist_dates = list(historical_dates)
    hist_prices = list(historical_prices)
    hist_demand = list(historical_demand)
    
    # Price plot
    ax1.plot(hist_dates, hist_prices, 'b-', label='Historical Price', linewidth=2, alpha=0.7)
    ax1.plot(prediction_dates, price_predictions, 'ro-', label='Predicted Price', linewidth=2, markersize=6)
    ax1.axvline(x=hist_dates[-1], color='gray', linestyle='--', alpha=0.5, label='Prediction Start')
    
    # Add trend line for historical data
    if len(hist_prices) > 7:
        z = np.polyfit(range(len(hist_prices[-7:])), hist_prices[-7:], 1)
        trend_line = np.polyval(z, range(len(hist_prices[-7:])))
        ax1.plot(hist_dates[-7:], trend_line, 'g--', alpha=0.5, label='Recent Trend')
    
    ax1.set_title(f'Paddy Price Prediction with Historical Context\n({start_date} to {end_date})', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (LKR/kg)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Demand plot
    ax2.plot(hist_dates, hist_demand, 'g-', label='Historical Demand', linewidth=2, alpha=0.7)
    ax2.plot(prediction_dates, demand_predictions, 'mo-', label='Predicted Demand', linewidth=2, markersize=6)
    ax2.axvline(x=hist_dates[-1], color='gray', linestyle='--', alpha=0.5, label='Prediction Start')
    
    # Add trend line for historical demand
    if len(hist_demand) > 7:
        z = np.polyfit(range(len(hist_demand[-7:])), hist_demand[-7:], 1)
        trend_line = np.polyval(z, range(len(hist_demand[-7:])))
        ax2.plot(hist_dates[-7:], trend_line, 'b--', alpha=0.5, label='Recent Trend')
    
    ax2.set_title(f'Demand Prediction with Historical Context\n({start_date} to {end_date})', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('Demand (Tons)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('predictions_with_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_combined_trend(prediction_dates, price_predictions, demand_predictions, start_date, end_date):
    """Plot combined price and demand trend"""
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Price on left axis
    color = 'red'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price (LKR/kg)', color=color, fontsize=12)
    line1 = ax1.plot(prediction_dates, price_predictions, 'ro-', linewidth=2, markersize=6, label='Price')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Demand on right axis
    ax2 = ax1.twinx()
    color = 'blue'
    ax2.set_ylabel('Demand (Tons)', color=color, fontsize=12)
    line2 = ax2.plot(prediction_dates, demand_predictions, 'bs-', linewidth=2, markersize=6, label='Demand')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title(f'Price vs Demand Prediction Trend\n({start_date} to {end_date})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('combined_trend.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==================== USER INPUT ====================

def get_prediction_dates():
    """Get prediction parameters from user"""
    print("\n📅 PREDICTION DATE SETUP")
    print("=" * 40)
    
    while True:
        try:
            start_date_str = input("Enter prediction start date (YYYY-MM-DD) or press Enter for tomorrow: ").strip()
            
            if start_date_str == "":
                start_date = pd.Timestamp.now() + pd.Timedelta(days=1)
            else:
                start_date = pd.to_datetime(start_date_str)
            
            n_days = input("Enter number of days to predict (default 7): ").strip()
            if n_days == "":
                n_days = 7
            else:
                n_days = int(n_days)
            
            if n_days <= 0:
                print("Please enter a positive number of days")
                continue
            
            end_date = start_date + pd.Timedelta(days=n_days - 1)
            
            print(f"\nPrediction Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({n_days} days)")
            confirm = input("Confirm? (y/n): ").strip().lower()
            
            if confirm in ['y', 'yes', '']:
                return start_date, n_days, end_date
            else:
                print("Let's try again...\n")
        
        except ValueError as e:
            print(f"Invalid input: {e}")
            print("Please try again with valid dates\n")
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again\n")


# ==================== MAIN PIPELINE ====================

def main():
    print("=" * 60)
    print("🌾 PADDY PRICE & DEMAND PREDICTION SYSTEM")
    print("=" * 60)
    
    # Get prediction parameters
    start_date, n_days, end_date = get_prediction_dates()
    
    print("\n📂 Loading models and data...")
    
    # Load models
    try:
        lstm = load_model("models/lstm_price_model_final.h5")
        print("✅ LSTM price model loaded")
    except Exception as e:
        print(f"❌ Error loading LSTM model: {e}")
        return
    
    try:
        xgb_model = joblib.load("models/xgb_demand_model_best_optimized.joblib")
        print("✅ XGBoost demand model loaded")
    except Exception as e:
        print(f"❌ Error loading XGBoost model: {e}")
        return
    
    # Load feature columns
    try:
        feature_cols_xgb = joblib.load("models/feature_columns_optimized.joblib")
        print(f"✅ Loaded {len(feature_cols_xgb)} demand features")
        
        feature_cols_lstm = joblib.load("models/lstm_feature_columns.joblib")
        print(f"✅ Loaded {len(feature_cols_lstm)} LSTM features")
        
        training_info = joblib.load("models/training_info.joblib")
        window_size = training_info.get('window_size', 21)
        print(f"✅ Training info loaded (window size: {window_size})")
        
    except Exception as e:
        print(f"⚠️ Could not load feature columns: {e}")
        window_size = 21
        return
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing dataset...")
    df = load_dataset()
    df_raw, df_mm, df_std, artifacts = preprocess(df, save_artifacts=False)
    
    # Apply feature engineering
    print("⚙️ Applying feature engineering...")
    df_mm = add_rolling_and_seasonal(df_mm)
    df_mm = add_price_momentum(df_mm)
    df_mm = df_mm.dropna().reset_index(drop=True)
    
    # Handle categorical columns
    cat_cols = [c for c in df_std.columns if df_std[c].dtype == 'object' and c != 'Date']
    for col in cat_cols:
        try:
            le = joblib.load(f"models/{col}_encoder.joblib")
            df_std[col] = le.transform(df_std[col])
        except:
            le = LabelEncoder()
            df_std[col] = le.fit_transform(df_std[col])
    
    df_std = add_lag_features(df_std, "Demand_Tons", n_lags=21)
    df_std = add_rolling_and_seasonal(df_std)
    df_std = add_price_momentum(df_std)
    df_std = df_std.dropna().reset_index(drop=True)
    
    # ========== PRICE PREDICTION ==========
    print(f"\n💰 Predicting prices for {n_days} days...")
    try:
        price_predictions, prediction_dates = predict_price_with_trend(
            df_mm, lstm, feature_cols_lstm, start_date, 
            n_steps=n_days, window_size=window_size
        )
        print("✅ Price predictions completed")
        
        # Display price predictions
        print("\n" + "=" * 50)
        print("PRICE PREDICTIONS")
        print("=" * 50)
        
        # Calculate statistics
        price_trend = np.polyfit(range(len(price_predictions)), price_predictions, 1)[0]
        
        for i, (date, price) in enumerate(zip(prediction_dates, price_predictions)):
            # Calculate day-over-day change
            if i > 0:
                change = price - price_predictions[i-1]
                change_pct = (change / price_predictions[i-1]) * 100
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"   {date.strftime('%Y-%m-%d')}: {price:.2f} LKR/kg  {arrow} {change:+.2f} ({change_pct:+.1f}%)")
            else:
                print(f"   {date.strftime('%Y-%m-%d')}: {price:.2f} LKR/kg")
        
        print(f"\n   📈 Trend: {'Increasing' if price_trend > 0 else 'Decreasing' if price_trend < 0 else 'Stable'}")
        print(f"   📊 Range: {min(price_predictions):.2f} - {max(price_predictions):.2f} LKR/kg")
        print(f"   📉 Volatility: {np.std(price_predictions):.4f}")
        
    except Exception as e:
        print(f"❌ Error in price prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== DEMAND PREDICTION ==========
    print(f"\n📦 Predicting demand for {n_days} days...")
    try:
        demand_predictions = predict_demand_with_trend(
            df_std, xgb_model, feature_cols_xgb, price_predictions, prediction_dates
        )
        print("✅ Demand predictions completed")
        
        # Display demand predictions
        print("\n" + "=" * 50)
        print("DEMAND PREDICTIONS")
        print("=" * 50)
        
        # Calculate statistics
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
        print(f"   📉 Volatility: {np.std(demand_predictions):.4f}")
        
    except Exception as e:
        print(f"❌ Error in demand prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 50)
    print("📋 PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"\n📊 Price:")
    print(f"   Average: {np.mean(price_predictions):.2f} LKR/kg")
    print(f"   Range: {min(price_predictions):.2f} - {max(price_predictions):.2f} LKR/kg")
    print(f"\n📊 Demand:")
    print(f"   Average: {np.mean(demand_predictions):.2f} Tons")
    print(f"   Range: {min(demand_predictions):.2f} - {max(demand_predictions):.2f} Tons")
    
    # Correlation between price and demand
    correlation = np.corrcoef(price_predictions, demand_predictions)[0, 1]
    print(f"\n📈 Price-Demand Correlation: {correlation:.3f}")
    
    # ========== VISUALIZATION ==========
    print("\n🎨 Generating visualizations...")
    
    # Get historical data for context
    historical_days = 60  # Show last 60 days of history
    historical_dates = pd.to_datetime(df_raw['Date'].iloc[-historical_days:])
    historical_prices = df_raw['Paddy_Price_LKR_per_kg'].iloc[-historical_days:].values
    historical_demand = df_raw['Demand_Tons'].iloc[-historical_days:].values
    
    # Plot predictions with historical context
    plot_predictions_with_history(
        historical_dates, historical_prices, historical_demand,
        prediction_dates, price_predictions, demand_predictions,
        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    )
    
    # Plot combined trend
    plot_combined_trend(
        prediction_dates, price_predictions, demand_predictions,
        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    )
    
    print("\n✅ Prediction completed successfully!")
    print("📁 Results saved as:")
    print("   - predictions_with_history.png")
    print("   - combined_trend.png")


if __name__ == "__main__":
    main()