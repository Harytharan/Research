# utils.py
import numpy as np
import pandas as pd

# ==================== FEATURE ENGINEERING FUNCTIONS ====================

def add_lag_features(df, target_col, n_lags=21):
    """Add lag features with proper handling"""
    df_copy = df.copy()
    for lag in range(1, n_lags + 1):
        df_copy[f"{target_col}_lag{lag}"] = df_copy[target_col].shift(lag)
    return df_copy


def add_rolling_and_seasonal(df):
    """Add rolling statistics and seasonal features"""
    df_copy = df.copy()
    
    # Convert Date to datetime if it's not
    if 'Date' in df_copy.columns:
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        
        # Time-based features
        df_copy['Month'] = df_copy['Date'].dt.month
        df_copy['Quarter'] = df_copy['Date'].dt.quarter
        df_copy['day_of_week'] = df_copy['Date'].dt.dayofweek
        df_copy['day_of_year'] = df_copy['Date'].dt.dayofyear
        
        # Cyclical encoding
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['Month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['Month'] / 12)
        df_copy['quarter_sin'] = np.sin(2 * np.pi * df_copy['Quarter'] / 4)
        df_copy['quarter_cos'] = np.cos(2 * np.pi * df_copy['Quarter'] / 4)
        df_copy['dow_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['dow_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['doy_sin'] = np.sin(2 * np.pi * df_copy['day_of_year'] / 365)
        df_copy['doy_cos'] = np.cos(2 * np.pi * df_copy['day_of_year'] / 365)
        df_copy['year_progress'] = df_copy['day_of_year'] / 365.0
        df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)
    
    # Rolling statistics with min_periods to handle edges
    for window in [3, 7, 14, 21]:
        if 'Demand_Tons' in df_copy.columns:
            df_copy[f'Demand_roll{window}'] = df_copy['Demand_Tons'].rolling(window, min_periods=1).mean()
        if 'Paddy_Price_LKR_per_kg' in df_copy.columns:
            df_copy[f'Price_roll{window}'] = df_copy['Paddy_Price_LKR_per_kg'].rolling(window, min_periods=1).mean()
        if 'Temperature_C' in df_copy.columns:
            df_copy[f'Temperature_roll{window}'] = df_copy['Temperature_C'].rolling(window, min_periods=1).mean()
        if 'Rainfall_mm' in df_copy.columns:
            df_copy[f'Rainfall_roll{window}'] = df_copy['Rainfall_mm'].rolling(window, min_periods=1).mean()
    
    # Rolling standard deviations
    if 'Demand_Tons' in df_copy.columns:
        df_copy['Demand_roll7_std'] = df_copy['Demand_Tons'].rolling(7, min_periods=1).std()
    if 'Paddy_Price_LKR_per_kg' in df_copy.columns:
        df_copy['Price_roll7_std'] = df_copy['Paddy_Price_LKR_per_kg'].rolling(7, min_periods=1).std()
    
    return df_copy


def add_price_momentum(df):
    """Add price momentum features"""
    df_copy = df.copy()
    
    if 'Paddy_Price_LKR_per_kg' in df_copy.columns:
        # Price differences
        df_copy['price_diff_1'] = df_copy['Paddy_Price_LKR_per_kg'].diff()
        df_copy['price_diff_3'] = df_copy['Paddy_Price_LKR_per_kg'].diff(3)
        df_copy['price_diff_7'] = df_copy['Paddy_Price_LKR_per_kg'].diff(7)
        
        # Price momentum (rate of change) - handle division by zero
        df_copy['price_momentum_3'] = df_copy['Paddy_Price_LKR_per_kg'].pct_change(3).replace([np.inf, -np.inf], 0)
        df_copy['price_momentum_7'] = df_copy['Paddy_Price_LKR_per_kg'].pct_change(7).replace([np.inf, -np.inf], 0)
        
        # Price volatility
        df_copy['price_volatility_7'] = df_copy['Paddy_Price_LKR_per_kg'].rolling(7, min_periods=1).std()
    
    return df_copy


def calculate_historical_trend(df, target_col, days=30):
    """Calculate historical trend statistics"""
    recent_data = df[target_col].tail(days)
    
    # Calculate trend metrics
    trend_direction = 1 if recent_data.is_monotonic_increasing else (-1 if recent_data.is_monotonic_decreasing else 0)
    trend_strength = recent_data.pct_change().mean() if len(recent_data) > 1 else 0
    
    # Calculate seasonal patterns
    if 'Date' in df.columns and len(df) > 0:
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy['month'] = df_copy['Date'].dt.month
        
        # Average by month for seasonality
        monthly_avg = df_copy.groupby('month')[target_col].mean()
        current_month = pd.to_datetime(df_copy['Date'].iloc[-1]).month
        seasonal_factor = monthly_avg[current_month] / monthly_avg.mean() if monthly_avg.mean() > 0 else 1.0
    else:
        seasonal_factor = 1.0
    
    return {
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'seasonal_factor': seasonal_factor,
        'recent_mean': recent_data.mean(),
        'recent_std': recent_data.std()
    }


def update_time_features(row, date):
    """Update time-based features in a row"""
    if isinstance(row, pd.Series):
        row['Month'] = date.month
        row['Quarter'] = date.quarter
        row['day_of_week'] = date.dayofweek
        row['day_of_year'] = date.dayofyear
        row['month_sin'] = np.sin(2 * np.pi * date.month / 12)
        row['month_cos'] = np.cos(2 * np.pi * date.month / 12)
        row['quarter_sin'] = np.sin(2 * np.pi * date.quarter / 4)
        row['quarter_cos'] = np.cos(2 * np.pi * date.quarter / 4)
        row['dow_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
        row['dow_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)
        row['doy_sin'] = np.sin(2 * np.pi * date.dayofyear / 365)
        row['doy_cos'] = np.cos(2 * np.pi * date.dayofyear / 365)
        row['year_progress'] = date.dayofyear / 365.0
        row['is_weekend'] = 1 if date.dayofweek >= 5 else 0
    return row