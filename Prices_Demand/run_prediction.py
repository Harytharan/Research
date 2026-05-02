# run_prediction.py
import numpy as np
import pandas as pd
import joblib


def add_time_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.0)

    def season_map(month):
        if month in [10, 11, 12, 1, 2, 3]:
            return "Maha"
        return "Yala"

    df["Season"] = df["Month"].apply(season_map)
    return df


def add_group_features(df):
    df = df.copy()
    group_cols = ["Region", "Rice_Type"]

    lag_base_cols = [
        "Paddy_Price_LKR_per_kg",
        "Demand_Tons",
        "Rainfall_mm",
        "Temperature_C",
        "Sentiment_Score",
        "News_Sentiment"
    ]

    for col in lag_base_cols:
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            df[f"{col}_lag{lag}"] = df.groupby(group_cols)[col].shift(lag)

    for col in ["Paddy_Price_LKR_per_kg", "Demand_Tons", "Rainfall_mm", "Temperature_C"]:
        for win in [3, 7, 14, 21, 30]:
            df[f"{col}_roll{win}"] = (
                df.groupby(group_cols)[col]
                .rolling(win)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )

    df["Price_diff_1"] = df.groupby(group_cols)["Paddy_Price_LKR_per_kg"].diff(1)
    df["Price_diff_3"] = df.groupby(group_cols)["Paddy_Price_LKR_per_kg"].diff(3)
    df["Price_diff_7"] = df.groupby(group_cols)["Paddy_Price_LKR_per_kg"].diff(7)

    df["Price_momentum_7"] = df.groupby(group_cols)["Paddy_Price_LKR_per_kg"].pct_change(7)
    df["Price_momentum_14"] = df.groupby(group_cols)["Paddy_Price_LKR_per_kg"].pct_change(14)

    df["Demand_diff_1"] = df.groupby(group_cols)["Demand_Tons"].diff(1)
    df["Demand_momentum_7"] = df.groupby(group_cols)["Demand_Tons"].pct_change(7)

    return df


def fill_numeric(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)
    return df


def calculate_seasonal_adjustment(df, target_col, window_days=30):
    """
    Calculate seasonal/cyclical adjustment pattern from historical data.
    Returns a list of adjustment factors based on day-of-year seasonality.
    """
    df_work = df.copy()
    df_work["Date"] = pd.to_datetime(df_work["Date"])
    
    if len(df_work) < window_days:
        return {}
    
    # Calculate trend using rolling mean
    df_work['Trend'] = df_work[target_col].rolling(window=window_days, center=True).mean()
    df_work['Detrended'] = df_work[target_col] - df_work['Trend']
    
    # Group by day of year to get seasonal pattern
    df_work['DayOfYear'] = df_work['Date'].dt.dayofyear
    seasonal_pattern = df_work.groupby('DayOfYear')['Detrended'].mean().to_dict()
    
    return seasonal_pattern


def apply_seasonal_adjustment(pred_value, pred_date, seasonal_pattern, boost_factor=1.0):
    """
    Apply historical seasonal pattern to prediction with optional upward boost.
    
    Args:
        pred_value: Raw prediction from model
        pred_date: Date of prediction
        seasonal_pattern: Historical seasonal factors
        boost_factor: Multiplier to boost predictions (1.05 = +5%, 1.08 = +8%, etc)
    """
    if not seasonal_pattern:
        return pred_value * boost_factor
    
    pred_date = pd.to_datetime(pred_date)
    day_of_year = pred_date.dayofyear
    
    # Find closest day in seasonal pattern if exact day not available
    if day_of_year in seasonal_pattern:
        seasonal_factor = seasonal_pattern[day_of_year]
    else:
        # Use nearest neighbor
        closest_day = min(seasonal_pattern.keys(), key=lambda x: abs(x - day_of_year))
        seasonal_factor = seasonal_pattern[closest_day]
    
    # Apply adjustment with dampening to prevent over-correction
    adjusted_value = pred_value + (seasonal_factor * 0.6)
    
    # Apply upward boost (e.g., boost_factor=1.05 adds 5%)
    adjusted_value = adjusted_value * boost_factor
    
    return adjusted_value


def _map_unknown_labels(values, known_classes, default_value):
    values = pd.Series(values).astype(str)
    known_set = set(known_classes)
    return values.apply(lambda v: v if v in known_set else default_value)


def _recent_diff_stats(series, window_size=30):
    values = pd.Series(series).dropna()
    if len(values) < 2:
        return 0.0, 0.0
    diffs = values.diff().dropna().tail(window_size)
    if len(diffs) == 0:
        return 0.0, 0.0
    return float(diffs.mean()), float(diffs.std())


def _apply_anchor_and_volatility(pred_value, last_value, step_idx, mean_diff, std_diff, anchor_pct=0.02):
    """
    Anchor early predictions close to last observed value and add small oscillations.
    """
    if last_value is None or np.isnan(last_value):
        return pred_value

    # Anchor the first prediction near the last observed value (slightly higher)
    if step_idx == 0:
        anchored = last_value * (1.0 + anchor_pct)
    else:
        anchored = pred_value

    # Add gentle oscillation based on recent diff volatility
    if std_diff > 0:
        # Deterministic, smooth oscillation
        oscillation = np.sin((step_idx + 1) * np.pi / 3.0) * std_diff * 0.4
        anchored = anchored + oscillation + (mean_diff * 0.3)

    return anchored


def _recent_diff_pattern(series, window_size=30):
    values = pd.Series(series).dropna()
    if len(values) < 2:
        return []
    diffs = values.diff().dropna().tail(window_size)
    if len(diffs) == 0:
        return []
    return diffs.tolist()


def _recent_centered_diff_pattern(series, window_size=14):
    values = pd.Series(series).dropna()
    if len(values) < 2:
        return []

    diffs = values.diff().dropna().tail(window_size)
    if len(diffs) == 0:
        return []

    centered = diffs - diffs.mean()
    return centered.tolist()


def _smooth_continuity(pred_value, prev_value, step_idx, mean_diff, std_diff, centered_pattern=None):
    """
    Keep forecast continuity realistic by blending early steps with recent trend
    and capping day-to-day jumps using recent volatility while preserving
    small ups/downs from recent historical movement.
    """
    if prev_value is None or np.isnan(prev_value):
        return pred_value

    expected_next = prev_value + (mean_diff * 0.65)

    # Strong continuity near the forecast start; relax after a few steps.
    if step_idx == 0:
        blend = 0.78
    elif step_idx == 1:
        blend = 0.58
    elif step_idx == 2:
        blend = 0.42
    else:
        blend = 0.22

    smoothed = (blend * expected_next) + ((1.0 - blend) * pred_value)

    # Inject micro-variation from recent centered diff pattern.
    if centered_pattern:
        smoothed += centered_pattern[step_idx % len(centered_pattern)] * 0.35

    # Gentle harmonic ripple so sequence is not perfectly monotonic.
    if std_diff > 0:
        smoothed += np.sin((step_idx + 1) * 1.7) * std_diff * 0.12

    # Cap abrupt daily movement.
    max_step = max(0.8, (std_diff * 1.5) + (abs(mean_diff) * 0.4))
    delta = np.clip(smoothed - prev_value, -max_step, max_step)

    # Avoid near-flat segments by enforcing a tiny minimum movement.
    min_move = max(0.05, std_diff * 0.06)
    if abs(delta) < min_move:
        direction = 1.0 if (step_idx % 3 != 1) else -1.0
        delta = direction * min_move

    return prev_value + delta


def add_encoded_columns(df):
    df = df.copy()

    region_encoder = joblib.load("models/region_encoder.joblib")
    rice_type_encoder = joblib.load("models/rice_type_encoder.joblib")
    season_encoder = joblib.load("models/season_encoder.joblib")

    # Map unseen labels to a safe default to avoid encoder errors
    default_region = region_encoder.classes_[0]
    default_rice = rice_type_encoder.classes_[0]
    default_season = season_encoder.classes_[0]

    df["Region"] = _map_unknown_labels(df["Region"], region_encoder.classes_, default_region)
    df["Rice_Type"] = _map_unknown_labels(df["Rice_Type"], rice_type_encoder.classes_, default_rice)
    df["Season"] = _map_unknown_labels(df["Season"], season_encoder.classes_, default_season)

    df["Region_encoded"] = region_encoder.transform(df["Region"])
    df["Rice_Type_encoded"] = rice_type_encoder.transform(df["Rice_Type"])
    df["Season_encoded"] = season_encoder.transform(df["Season"])

    return df


def predict_price_with_trend(df_mm, lstm, feature_cols_lstm, start_date, n_steps=7, window_size=30, boost_factor=1.06):
    """
    Predict paddy prices with trend and seasonal patterns.
    
    Args:
        boost_factor: Multiplier to boost predictions above baseline (default 1.06 = +6%)
    """
    feature_scaler = joblib.load("models/lstm_feature_scaler.joblib")
    target_scaler = joblib.load("models/lstm_target_scaler.joblib")

    df_work = df_mm.copy()
    df_work["Date"] = pd.to_datetime(df_work["Date"])
    df_work = df_work.sort_values("Date").reset_index(drop=True)

    if len(df_work) < window_size:
        raise ValueError(f"Not enough rows for LSTM window size {window_size}. Found {len(df_work)} rows.")

    # Calculate seasonal pattern from historical data
    seasonal_pattern = calculate_seasonal_adjustment(df_work, "Paddy_Price_LKR_per_kg", window_days=30)
    last_price = float(df_work["Paddy_Price_LKR_per_kg"].iloc[-1]) if len(df_work) > 0 else None
    mean_diff, std_diff = _recent_diff_stats(df_work["Paddy_Price_LKR_per_kg"], window_size=30)
    recent_pattern = _recent_diff_pattern(df_work["Paddy_Price_LKR_per_kg"], window_size=14)
    centered_pattern = _recent_centered_diff_pattern(df_work["Paddy_Price_LKR_per_kg"], window_size=14)
    prev_price = last_price
    
    price_predictions = []
    prediction_dates = []
    current_date = pd.to_datetime(start_date)

    for step_idx in range(n_steps):
        new_row = df_work.iloc[-1].copy()
        new_row["Date"] = current_date

        df_work = pd.concat([df_work, pd.DataFrame([new_row])], ignore_index=True)

        df_work = add_time_features(df_work)
        df_work = add_group_features(df_work)
        df_work = fill_numeric(df_work)
        df_work = add_encoded_columns(df_work)

        seq_df = df_work.tail(window_size).copy()
        X_input = seq_df[feature_cols_lstm].copy()
        X_input = feature_scaler.transform(X_input)
        X_input = np.array([X_input], dtype=np.float32)

        pred_scaled = lstm.predict(X_input, verbose=0)
        pred_price = float(target_scaler.inverse_transform(pred_scaled)[0][0])
        
        # Apply seasonal adjustment with upward boost
        pred_price = apply_seasonal_adjustment(pred_price, current_date, seasonal_pattern, boost_factor=boost_factor)
        pred_price = _apply_anchor_and_volatility(
            pred_price,
            last_price,
            step_idx,
            mean_diff,
            std_diff,
            anchor_pct=0.004
        )

        # Blend with recent diff pattern to ensure small ups/downs
        if recent_pattern:
            pattern_delta = recent_pattern[step_idx % len(recent_pattern)]
            pred_price = pred_price + (pattern_delta * 0.25)

        pred_price = _smooth_continuity(
            pred_price,
            prev_price,
            step_idx,
            mean_diff,
            std_diff,
            centered_pattern=centered_pattern
        )

        df_work.loc[df_work.index[-1], "Paddy_Price_LKR_per_kg"] = pred_price
        prev_price = pred_price

        price_predictions.append(pred_price)
        prediction_dates.append(current_date)

        current_date += pd.Timedelta(days=1)

    return price_predictions, prediction_dates


def predict_demand_with_trend(df_std, xgb, feature_cols_xgb, price_predictions, prediction_dates, boost_factor=1.08):
    """
    Predict paddy demand with trend and seasonal patterns.
    
    Args:
        boost_factor: Multiplier to boost predictions above baseline (default 1.08 = +8%)
    """
    df_work = df_std.copy()
    df_work["Date"] = pd.to_datetime(df_work["Date"])
    df_work = df_work.sort_values("Date").reset_index(drop=True)

    # Calculate seasonal pattern from historical data
    seasonal_pattern = calculate_seasonal_adjustment(df_work, "Demand_Tons", window_days=30)
    last_demand = float(df_work["Demand_Tons"].iloc[-1]) if len(df_work) > 0 else None
    mean_diff, std_diff = _recent_diff_stats(df_work["Demand_Tons"], window_size=30)
    recent_pattern = _recent_diff_pattern(df_work["Demand_Tons"], window_size=14)
    centered_pattern = _recent_centered_diff_pattern(df_work["Demand_Tons"], window_size=14)
    prev_demand = last_demand

    demand_predictions = []

    for step_idx, (pred_price, pred_date) in enumerate(zip(price_predictions, prediction_dates)):
        new_row = df_work.iloc[-1].copy()
        new_row["Date"] = pd.to_datetime(pred_date)
        new_row["Paddy_Price_LKR_per_kg"] = float(pred_price)
        new_row["Price_LSTM_pred"] = float(pred_price)

        df_work = pd.concat([df_work, pd.DataFrame([new_row])], ignore_index=True)

        df_work = add_time_features(df_work)
        df_work = add_group_features(df_work)
        df_work = fill_numeric(df_work)
        df_work = add_encoded_columns(df_work)

        latest_row = df_work.tail(1).copy()
        X_input = latest_row[feature_cols_xgb].copy()
        pred_demand = float(xgb.predict(X_input)[0])
        
        # Apply seasonal adjustment with upward boost
        pred_demand = apply_seasonal_adjustment(pred_demand, pred_date, seasonal_pattern, boost_factor=boost_factor)
        pred_demand = _apply_anchor_and_volatility(
            pred_demand,
            last_demand,
            step_idx,
            mean_diff,
            std_diff,
            anchor_pct=0.004
        )

        # Blend with recent diff pattern to ensure small ups/downs
        if recent_pattern:
            pattern_delta = recent_pattern[step_idx % len(recent_pattern)]
            pred_demand = pred_demand + (pattern_delta * 0.25)

        pred_demand = _smooth_continuity(
            pred_demand,
            prev_demand,
            step_idx,
            mean_diff,
            std_diff,
            centered_pattern=centered_pattern
        )

        df_work.loc[df_work.index[-1], "Demand_Tons"] = pred_demand
        prev_demand = pred_demand
        demand_predictions.append(pred_demand)

    return demand_predictions