# utils/prediction_utils.py
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

from load import load_dataset
from pre_process import preprocess
from utils import (
    add_lag_features, add_rolling_and_seasonal, add_price_momentum,
    calculate_historical_trend, update_time_features
)

def load_models():
    """Load all trained models"""
    models = {}
    
    try:
        models['lstm'] = load_model("models/lstm_price_model_final.h5")
        models['xgb'] = joblib.load("models/xgb_demand_model_best_optimized.joblib")
        models['xgb_features'] = joblib.load("models/feature_columns_optimized.joblib")
        models['lstm_features'] = joblib.load("models/lstm_feature_columns.joblib")
        
        training_info = joblib.load("models/training_info.joblib")
        models['window_size'] = training_info.get('window_size', 21)
        
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def load_and_preprocess_data():
    """Load and preprocess historical data"""
    df = load_dataset()
    df_raw, df_mm, df_std, _ = preprocess(df, save_artifacts=False)
    
    # Apply feature engineering
    df_mm = add_rolling_and_seasonal(df_mm)
    df_mm = add_price_momentum(df_mm)
    df_mm = df_mm.dropna().reset_index(drop=True)
    
    return df_raw, df_mm, df_std

def predict_price_with_trend(df, lstm_model, feature_cols, start_date, n_steps=7, window_size=21):
    """Predict future prices while maintaining historical trends"""
    
    df_extended = df.copy()
    predictions = []
    
    # Calculate historical trends
    price_trend = calculate_historical_trend(df_extended, 'Paddy_Price_LKR_per_kg', days=60)
    
    # Get recent price history
    recent_prices = df_extended['Paddy_Price_LKR_per_kg'].tail(30).values
    
    # Get the last window of data
    current_data = df_extended[feature_cols].values[-window_size:]
    
    for i in range(n_steps):
        sequence = current_data.reshape(1, window_size, len(feature_cols))
        lstm_pred = lstm_model.predict(sequence, verbose=0)[0][0]
        
        # Blend with trend
        if len(recent_prices) > 0:
            if i == 0:
                last_price = recent_prices[-1]
            else:
                last_price = predictions[-1]
            
            trend_component = last_price * (1 + price_trend['trend_strength'])
            trend_component *= price_trend['seasonal_factor']
            
            blended_price = 0.7 * lstm_pred + 0.3 * trend_component
        else:
            blended_price = lstm_pred
        
        # Apply constraints
        historical_mean = df_extended['Paddy_Price_LKR_per_kg'].tail(90).mean()
        historical_std = df_extended['Paddy_Price_LKR_per_kg'].tail(90).std()
        
        max_deviation = 2 * historical_std
        min_allowed = max(historical_mean - max_deviation, df_extended['Paddy_Price_LKR_per_kg'].min() * 0.8)
        max_allowed = min(historical_mean + max_deviation, df_extended['Paddy_Price_LKR_per_kg'].max() * 1.2)
        
        blended_price = np.clip(blended_price, min_allowed, max_allowed)
        predictions.append(blended_price)
        
        # Update for next iteration
        new_date = start_date + pd.Timedelta(days=i)
        new_features = create_future_features(df_extended, blended_price, new_date, feature_cols)
        current_data = np.vstack([current_data[1:], new_features])
        
        new_row = df_extended.iloc[-1].copy()
        new_row['Paddy_Price_LKR_per_kg'] = blended_price
        new_row['Date'] = new_date
        new_row = update_time_features(new_row, new_date)
        
        df_extended = pd.concat([df_extended, pd.DataFrame([new_row])], ignore_index=True)
        recent_prices = np.append(recent_prices[1:], blended_price) if len(recent_prices) > 1 else np.array([blended_price])
    
    prediction_dates = [start_date + pd.Timedelta(days=i) for i in range(n_steps)]
    return predictions, prediction_dates

def create_future_features(df, predicted_price, date, feature_cols):
    """Create feature vector for future prediction"""
    if len(df) > 0:
        new_features = df[feature_cols].iloc[-1].copy()
    else:
        new_features = np.zeros(len(feature_cols))
    
    if 'Paddy_Price_LKR_per_kg' in feature_cols:
        price_idx = feature_cols.index('Paddy_Price_LKR_per_kg')
        new_features[price_idx] = predicted_price
    
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

def predict_demand_with_trend(df_original, xgb_model, feature_columns, price_predictions, prediction_dates):
    """Predict demand while maintaining historical trends"""
    
    future_df = create_future_dataframe(df_original, price_predictions, prediction_dates)
    demand_trend = calculate_historical_trend(df_original, 'Demand_Tons', days=60)
    
    X_future = prepare_demand_features(future_df, feature_columns)
    xgb_predictions = xgb_model.predict(X_future)
    
    trend_predictions = []
    recent_demand = df_original['Demand_Tons'].tail(30).values
    
    for i, xgb_pred in enumerate(xgb_predictions):
        if len(recent_demand) > 0:
            last_demand = recent_demand[-1] if i == 0 else trend_predictions[-1]
            trend_component = last_demand * (1 + demand_trend['trend_strength']) * demand_trend['seasonal_factor']
            blended_demand = 0.6 * xgb_pred + 0.4 * trend_component
        else:
            blended_demand = xgb_pred
        
        # Apply constraints
        historical_mean = df_original['Demand_Tons'].tail(90).mean()
        historical_std = df_original['Demand_Tons'].tail(90).std()
        
        max_deviation = 2 * historical_std
        min_allowed = max(historical_mean - max_deviation, df_original['Demand_Tons'].min() * 0.8)
        max_allowed = min(historical_mean + max_deviation, df_original['Demand_Tons'].max() * 1.2)
        
        blended_demand = np.clip(blended_demand, min_allowed, max_allowed)
        trend_predictions.append(blended_demand)
        recent_demand = np.append(recent_demand[1:], blended_demand) if len(recent_demand) > 1 else np.array([blended_demand])
    
    return np.array(trend_predictions)

def create_future_dataframe(df_original, price_predictions, prediction_dates):
    """Create dataframe for future predictions"""
    future_data = []
    last_row = df_original.iloc[-1].copy()
    
    for price, date in zip(price_predictions, prediction_dates):
        new_row = last_row.copy()
        new_row['Date'] = date
        new_row['Paddy_Price_LKR_per_kg'] = price
        new_row = update_time_features(new_row, date)
        future_data.append(new_row)
    
    future_df = pd.DataFrame(future_data)
    future_df = add_lag_features(future_df, "Demand_Tons", n_lags=21)
    future_df = add_rolling_and_seasonal(future_df)
    future_df = add_price_momentum(future_df)
    
    return future_df

def prepare_demand_features(df_future, feature_columns):
    """Prepare features for demand prediction"""
    df_filled = df_future.copy()
    
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(0)
    
    for col in feature_columns:
        if col not in df_filled.columns:
            df_filled[col] = 0
    
    return df_filled[feature_columns]