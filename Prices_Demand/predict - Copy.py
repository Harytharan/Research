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
from featureeng import create_lstm_sequences


# Enhanced Feature Engineering Functions
def add_lag_features(df, target_col, n_lags=21):
    df_copy = df.copy()
    for lag in range(1, n_lags + 1):
        df_copy[f"{target_col}_lag{lag}"] = df_copy[target_col].shift(lag)
    return df_copy


def add_rolling_and_seasonal(df):
    df_copy = df.copy()

    # Multiple rolling windows
    for window in [3, 7, 14, 21]:
        df_copy[f'Demand_roll{window}'] = df_copy['Demand_Tons'].rolling(window).mean()
        df_copy[f'Price_roll{window}'] = df_copy['Paddy_Price_LKR_per_kg'].rolling(window).mean()
        df_copy[f'Temperature_roll{window}'] = df_copy['Temperature_C'].rolling(window).mean()
        df_copy[f'Rainfall_roll{window}'] = df_copy['Rainfall_mm'].rolling(window).mean()

    # Rolling statistics
    df_copy['Demand_roll7_std'] = df_copy['Demand_Tons'].rolling(7).std()
    df_copy['Price_roll7_std'] = df_copy['Paddy_Price_LKR_per_kg'].rolling(7).std()

    # Enhanced seasonal encoding
    df_copy['Month'] = pd.to_datetime(df_copy['Date']).dt.month
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['Month'] / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['Month'] / 12)

    # Quarter encoding
    df_copy['Quarter'] = pd.to_datetime(df_copy['Date']).dt.quarter
    df_copy['quarter_sin'] = np.sin(2 * np.pi * df_copy['Quarter'] / 4)
    df_copy['quarter_cos'] = np.cos(2 * np.pi * df_copy['Quarter'] / 4)

    # Day-of-week encoding
    df_copy['day_of_week'] = pd.to_datetime(df_copy['Date']).dt.dayofweek
    df_copy['dow_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
    df_copy['dow_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)

    # Day of year encoding
    df_copy['day_of_year'] = pd.to_datetime(df_copy['Date']).dt.dayofyear
    df_copy['doy_sin'] = np.sin(2 * np.pi * df_copy['day_of_year'] / 365)
    df_copy['doy_cos'] = np.cos(2 * np.pi * df_copy['day_of_year'] / 365)

    # Year progression
    df_copy['year_progress'] = df_copy['day_of_year'] / 365.0

    # Weekend flag
    df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)

    return df_copy


def add_price_momentum(df):
    df_copy = df.copy()

    # Price differences
    df_copy['price_diff_1'] = df_copy['Paddy_Price_LKR_per_kg'].diff()
    df_copy['price_diff_3'] = df_copy['Paddy_Price_LKR_per_kg'].diff(3)
    df_copy['price_diff_7'] = df_copy['Paddy_Price_LKR_per_kg'].diff(7)

    # Price momentum
    df_copy['price_momentum_3'] = df_copy['Paddy_Price_LKR_per_kg'].pct_change(3)
    df_copy['price_momentum_7'] = df_copy['Paddy_Price_LKR_per_kg'].pct_change(7)

    # Price volatility
    df_copy['price_volatility_7'] = df_copy['Paddy_Price_LKR_per_kg'].rolling(7).std()

    return df_copy


# Enhanced Future Price Prediction
def predict_price_future_enhanced(df, lstm_model, feature_cols, start_date, n_steps=7, window_size=21):

    df_extended = df.copy()
    predictions = []

    # Get the last window of data
    current_data = df_extended[feature_cols].values[-window_size:]

    for i in range(n_steps):
        sequence = current_data.reshape(1, window_size, len(feature_cols))
        pred_price = lstm_model.predict(sequence, verbose=0)[0][0]
        predictions.append(pred_price)
        new_features = create_enhanced_future_features(df_extended, pred_price, start_date + pd.Timedelta(days=i),
                                                       feature_cols)
        current_data = np.vstack([current_data[1:], new_features])

        new_row = df_extended.iloc[-1].copy()
        new_row['Paddy_Price_LKR_per_kg'] = pred_price
        new_row['Date'] = start_date + pd.Timedelta(days=i)

        # Update all time-based features in the new row
        new_row = update_time_features(new_row, start_date + pd.Timedelta(days=i))

        df_extended = pd.concat([df_extended, pd.DataFrame([new_row])], ignore_index=True)

        # Recalculate rolling features for the extended dataframe
        if len(df_extended) > 7:
            df_extended = recalculate_rolling_features(df_extended)

    prediction_dates = [start_date + pd.Timedelta(days=i) for i in range(n_steps)]
    return predictions, prediction_dates


def create_enhanced_future_features(df, predicted_price, date, feature_cols):

    # Start with the last row's features
    new_features = df[feature_cols].iloc[-1].copy()

    if 'Paddy_Price_LKR_per_kg' in feature_cols:
        price_idx = feature_cols.index('Paddy_Price_LKR_per_kg')
        new_features[price_idx] = predicted_price

    if 'Month' in feature_cols:
        month_idx = feature_cols.index('Month')
        new_features[month_idx] = date.month

    if 'day_of_week' in feature_cols:
        dow_idx = feature_cols.index('day_of_week')
        new_features[dow_idx] = date.dayofweek

    if 'Quarter' in feature_cols:
        quarter_idx = feature_cols.index('Quarter')
        new_features[quarter_idx] = date.quarter

    if 'day_of_year' in feature_cols:
        doy_idx = feature_cols.index('day_of_year')
        new_features[doy_idx] = date.dayofyear

    seasonal_updates = {
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

    for feature_name, value in seasonal_updates.items():
        if feature_name in feature_cols:
            idx = feature_cols.index(feature_name)
            new_features[idx] = value

    # Update price
    if 'price_diff_1' in feature_cols:
        diff_idx = feature_cols.index('price_diff_1')
        # Use the difference
        if len(df) > 1:
            last_price = df['Paddy_Price_LKR_per_kg'].iloc[-1]
            new_features[diff_idx] = predicted_price - last_price

    return new_features.values


def update_time_features(row, date):
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


def recalculate_rolling_features(df):
    df_copy = df.copy()

    # Recalculate rolling means
    for window in [3, 7, 14, 21]:
        if f'Demand_roll{window}' in df_copy.columns:
            df_copy[f'Demand_roll{window}'] = df_copy['Demand_Tons'].rolling(window, min_periods=1).mean()
        if f'Price_roll{window}' in df_copy.columns:
            df_copy[f'Price_roll{window}'] = df_copy['Paddy_Price_LKR_per_kg'].rolling(window, min_periods=1).mean()

    # Recalculate rolling std
    if 'Demand_roll7_std' in df_copy.columns:
        df_copy['Demand_roll7_std'] = df_copy['Demand_Tons'].rolling(7, min_periods=1).std()
    if 'Price_roll7_std' in df_copy.columns:
        df_copy['Price_roll7_std'] = df_copy['Paddy_Price_LKR_per_kg'].rolling(7, min_periods=1).std()

    # Recalculate price momentum features
    df_copy = add_price_momentum(df_copy)

    return df_copy


#  Enhanced Future Demand Prediction
def create_enhanced_future_dataframe(df_original, price_predictions, prediction_dates):
    future_data = []

    for i, (price, date) in enumerate(zip(price_predictions, prediction_dates)):
        # Create base row with the last available values
        base_row = df_original.iloc[-1].copy()

        # Update date
        base_row['Date'] = date

        # Update with predicted price
        base_row['Price_LSTM_pred'] = price

        # Update all time-based features
        base_row = update_time_features(base_row, date)

        future_data.append(base_row)

    future_df = pd.DataFrame(future_data)

    # Add enhanced features
    future_df = add_lag_features(future_df, "Demand_Tons", n_lags=21)
    future_df = add_rolling_and_seasonal(future_df)
    future_df = add_price_momentum(future_df)

    return future_df


def prepare_enhanced_demand_features(df_future, feature_columns):

    # Fill NaN values using forward fill and then backward fill
    df_filled = df_future.copy()

    # For numerical columns
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(method='ffill').fillna(method='bfill')

    # If still NaN, fill with column mean
    df_filled[numeric_cols] = df_filled[numeric_cols].apply(lambda x: x.fillna(x.mean()) if x.notna().any() else x)

    # Ensure we have all required columns
    missing_cols = set(feature_columns) - set(df_filled.columns)
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        for col in missing_cols:
            df_filled[col] = 0

    # Select and order columns exactly as during training
    X_future = df_filled[feature_columns]

    return X_future


def predict_demand_future_enhanced(df_original, xgb_model, feature_columns, price_predictions, prediction_dates):

    # Create enhanced future dataframe
    future_df = create_enhanced_future_dataframe(df_original, price_predictions, prediction_dates)

    # Prepare features for prediction
    X_future = prepare_enhanced_demand_features(future_df, feature_columns)

    print(f"🔧 Prepared demand features shape: {X_future.shape}")
    print(f"🔧 Features used: {len(feature_columns)}")

    # Predict demand
    future_demand = xgb_model.predict(X_future)

    return future_demand


#  Visualization Functions
def plot_enhanced_predictions(historical_dates, historical_prices, historical_demand,
                              prediction_dates, price_predictions, demand_predictions,
                              start_date, end_date):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Convert to lists to avoid indexing issues
    hist_dates_list = list(historical_dates)
    hist_prices_list = list(historical_prices)
    hist_demand_list = list(historical_demand)

    #  Price predictions
    ax1.plot(hist_dates_list, hist_prices_list, 'b-', label='Historical Price', linewidth=2, alpha=0.8)
    ax1.plot(prediction_dates, price_predictions, 'ro-', label='Predicted Price', linewidth=2, markersize=6)

    # Add vertical line at the start of prediction
    if hist_dates_list:
        ax1.axvline(x=hist_dates_list[-1], color='gray', linestyle='--', alpha=0.7, label='Prediction Start')

    ax1.set_title(f'Paddy Price Prediction ({start_date} to {end_date})', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price (LKR/kg)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Add price statistics
    price_range = max(price_predictions) - min(price_predictions)
    ax1.text(0.02, 0.98, f'Price Range: {price_range:.2f} LKR/kg',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Demand predictions
    ax2.plot(hist_dates_list, hist_demand_list, 'g-', label='Historical Demand', linewidth=2, alpha=0.8)
    ax2.plot(prediction_dates, demand_predictions, 'mo-', label='Predicted Demand', linewidth=2, markersize=6)

    # Add vertical line at the start of prediction
    if hist_dates_list:
        ax2.axvline(x=hist_dates_list[-1], color='gray', linestyle='--', alpha=0.7, label='Prediction Start')

    ax2.set_title(f'Demand Prediction ({start_date} to {end_date})', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Demand (Tons)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # Add demand statistics
    demand_range = max(demand_predictions) - min(demand_predictions)
    ax2.text(0.02, 0.98, f'Demand Range: {demand_range:.2f} Tons',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('enhanced_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_combined_trend_enhanced(prediction_dates, price_predictions, demand_predictions, start_date, end_date):

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Price on left axis
    color = 'red'
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price (LKR/kg)', color=color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(prediction_dates, price_predictions, 'ro-', linewidth=2, markersize=6, label='Price')
    ax1.tick_params(axis='y', labelcolor=color)

    # Demand on right axis
    ax2 = ax1.twinx()
    color = 'blue'
    ax2.set_ylabel('Demand (Tons)', color=color, fontsize=12, fontweight='bold')
    line2 = ax2.plot(prediction_dates, demand_predictions, 'bs-', linewidth=2, markersize=6, label='Demand')
    ax2.tick_params(axis='y', labelcolor=color)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f'Price vs Demand Prediction Trend ({start_date} to {end_date})',
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('enhanced_combined_trend.png', dpi=300, bbox_inches='tight')
    plt.show()


# User Input Function
def get_prediction_dates():
    print("\nPREDICTION DATE SETUP")
    print("=" * 40)

    while True:
        try:
            start_date_str = input("Enter prediction start date (YYYY-MM-DD) or press Enter for tomorrow: ").strip()

            if start_date_str == "":
                # Default to tomorrow
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

            print(
                f"\nPrediction Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({n_days} days)")
            confirm = input("Confirm? (y/n): ").strip().lower()

            if confirm in ['y', 'yes', '']:
                return start_date, n_days, end_date
            else:
                print("Let's try again...\n")

        except ValueError as e:
            print(f"Invalid input: {e}")
            print("Please try again with valid dates (YYYY-MM-DD format)\n")
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again\n")


# Enhanced Main Prediction Pipeline
def main():
    print("Enhanced Paddy Price & Demand Prediction System")
    print("=" * 55)

    # Get prediction dates from user
    start_date, n_days, end_date = get_prediction_dates()

    print("\nLoading enhanced models and data...")

    # Load enhanced models
    try:
        lstm = load_model("models/lstm_price_model_final.h5")
        print("Enhanced LSTM model loaded")
    except:
        print("Could not load LSTM model")
        return

    try:
        model_xgb = joblib.load("models/xgb_demand_model_best_optimized.joblib")
        print("Enhanced XGBoost model loaded")
    except:
        print("Could not load XGBoost model")
        return

    # Load feature columns and training info
    try:
        feature_columns_xgb = joblib.load("models/feature_columns_optimized.joblib")
        print(f"Feature columns loaded: {len(feature_columns_xgb)} features")

        training_info = joblib.load("models/training_info.joblib")
        window_size = training_info.get('window_size', 21)
        print(f"Training info loaded - Window size: {window_size}")

    except Exception as e:
        print(f"Could not load feature columns or training info: {e}")
        window_size = 21
        return

    # Load and preprocess data
    print("Loading and preprocessing dataset...")
    df = load_dataset()
    df_raw, df_mm, df_std, artifacts = preprocess(df, save_artifacts=False, artifact_dir="models")

    # Apply enhanced feature engineering
    print("Applying enhanced feature engineering...")
    df_mm = add_rolling_and_seasonal(df_mm)
    df_mm = add_price_momentum(df_mm)
    df_mm = df_mm.dropna().reset_index(drop=True)

    df_std_eng = add_lag_features(df_std, "Demand_Tons", n_lags=21)
    df_std_eng = add_rolling_and_seasonal(df_std_eng)
    df_std_eng = add_price_momentum(df_std_eng)
    df_std_eng = df_std_eng.dropna().reset_index(drop=True)

    # Load categorical encoders if they exist
    cat_cols = [c for c in df_std_eng.columns if df_std_eng[c].dtype == 'object' and c != 'Date']
    for col in cat_cols:
        try:
            le = joblib.load(f"models/{col}_encoder.joblib")
            df_std_eng[col] = le.transform(df_std_eng[col])
        except:
            le = LabelEncoder()
            df_std_eng[col] = le.fit_transform(df_std_eng[col])

    print(f"Enhanced processed data shape: {df_std_eng.shape}")

    # Get LSTM feature columns
    try:
        feature_cols_lstm = joblib.load("models/lstm_feature_columns.joblib")
        print(f"LSTM feature columns loaded: {len(feature_cols_lstm)} features")
    except:
        exclude = {"Date", "Paddy_Price_LKR_per_kg", "Demand_Tons"}
        feature_cols_lstm = [c for c in df_mm.columns if c not in exclude]
        print(f"Using {len(feature_cols_lstm)} features for LSTM prediction")

    print(f"LSTM features: {feature_cols_lstm}")

    # Predict future prices using enhanced method
    print(f"\nPredicting prices for {n_days} days starting {start_date.strftime('%Y-%m-%d')}...")
    try:
        price_predictions, prediction_dates = predict_price_future_enhanced(
            df_mm, lstm, feature_cols_lstm, start_date, n_steps=n_days, window_size=window_size
        )
        print("Enhanced price predictions completed")

    except Exception as e:
        print(f"Enhanced prediction failed: {e}")
        return

    # Display price predictions with analysis
    print("\nENHANCED PRICE PREDICTIONS:")
    print("-" * 45)
    price_variation = np.std(price_predictions)
    for i, (date, price) in enumerate(zip(prediction_dates, price_predictions), 1):
        trend = "↑" if i > 1 and price > price_predictions[i - 2] else "↓" if i > 1 and price < price_predictions[
            i - 2] else "→"
        print(f"   {date.strftime('%Y-%m-%d')}: {price:.2f} LKR/kg {trend}")

    print(f"\nPrice Variation Analysis:")
    print(f"   Standard Deviation: {price_variation:.4f}")
    print(f"   Price Range: {min(price_predictions):.2f} - {max(price_predictions):.2f} LKR/kg")

    # Predict future demand using enhanced method
    print(f"\nPredicting demand for {n_days} days...")
    try:
        demand_predictions = predict_demand_future_enhanced(
            df_std_eng, model_xgb, feature_columns_xgb, price_predictions, prediction_dates
        )
        print("Enhanced demand predictions completed")

        # Display demand predictions
        print("\nENHANCED DEMAND PREDICTIONS:")
        print("-" * 45)
        demand_variation = np.std(demand_predictions)
        for i, (date, demand) in enumerate(zip(prediction_dates, demand_predictions), 1):
            trend = "↑" if i > 1 and demand > demand_predictions[i - 2] else "↓" if i > 1 and demand < \
                                                                                    demand_predictions[i - 2] else "→"
            print(f"   {date.strftime('%Y-%m-%d')}: {demand:.2f} Tons {trend}")

        print(f"\nDemand Variation Analysis:")
        print(f"   Standard Deviation: {demand_variation:.4f}")
        print(f"   Demand Range: {min(demand_predictions):.2f} - {max(demand_predictions):.2f} Tons")

    except Exception as e:
        print(f"Error in demand prediction: {e}")
        import traceback
        traceback.print_exc()
        return

    # Enhanced summary
    print("\nENHANCED PREDICTION SUMMARY")
    print("=" * 55)
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Average Price: {np.mean(price_predictions):.2f} LKR/kg")
    print(f"Average Demand: {np.mean(demand_predictions):.2f} Tons")
    print(f"Price Range: {min(price_predictions):.2f} - {max(price_predictions):.2f} LKR/kg")
    print(f"Demand Range: {min(demand_predictions):.2f} - {max(demand_predictions):.2f} Tons")
    print(f"Price Volatility: {np.std(price_predictions):.4f}")
    print(f"Demand Volatility: {np.std(demand_predictions):.4f}")

    # Create enhanced visualizations
    print("\nGenerating enhanced visualizations...")

    # Get historical data for plotting
    historical_cutoff = min(90, len(df_raw))  # Increased to 90 days for better context
    historical_dates = pd.to_datetime(df_raw['Date'].iloc[-historical_cutoff:])
    historical_prices = df_raw['Paddy_Price_LKR_per_kg'].iloc[-historical_cutoff:].values
    historical_demand = df_raw['Demand_Tons'].iloc[-historical_cutoff:].values

    # Plot enhanced predictions
    plot_enhanced_predictions(
        historical_dates, historical_prices, historical_demand,
        prediction_dates, price_predictions, demand_predictions,
        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    )

    # Plot enhanced combined trend
    plot_combined_trend_enhanced(
        prediction_dates, price_predictions, demand_predictions,
        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    )

    print("\nEnhanced prediction pipeline completed!")
    print("Results saved as 'enhanced_prediction_results.png' and 'enhanced_combined_trend.png'")
    print("Check the graphs for detailed visualization of price and demand trends!")


if __name__ == "__main__":
    main()