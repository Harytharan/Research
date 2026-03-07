import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from load import load_dataset
from pre_process import preprocess
from featureeng import create_lstm_sequences


#  Enhanced LSTM
def build_enhanced_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64, return_sequences=True,
             kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(32, return_sequences=False,
             kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.1),
        Dense(1, activation="linear")
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mape"])
    return model


# - Lag Features
def add_lag_features(df, target_col, n_lags=14):
    df_copy = df.copy()
    for lag in range(1, n_lags + 1):
        df_copy[f"{target_col}_lag{lag}"] = df_copy[target_col].shift(lag)
    return df_copy


# Add Rolling
def add_rolling_and_seasonal(df):
    df_copy = df.copy()

    # Multiple
    for window in [3, 7, 14, 21]:
        df_copy[f'Demand_roll{window}'] = df_copy['Demand_Tons'].rolling(window).mean()
        df_copy[f'Price_roll{window}'] = df_copy['Paddy_Price_LKR_per_kg'].rolling(window).mean()
        df_copy[f'Temperature_roll{window}'] = df_copy['Temperature_C'].rolling(window).mean()
        df_copy[f'Rainfall_roll{window}'] = df_copy['Rainfall_mm'].rolling(window).mean()

    #  statistics
    df_copy['Demand_roll7_std'] = df_copy['Demand_Tons'].rolling(7).std()
    df_copy['Price_roll7_std'] = df_copy['Paddy_Price_LKR_per_kg'].rolling(7).std()

    # Month encoding
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

    # Day of year encoding for seasonal patterns
    df_copy['day_of_year'] = pd.to_datetime(df_copy['Date']).dt.dayofyear
    df_copy['doy_sin'] = np.sin(2 * np.pi * df_copy['day_of_year'] / 365)
    df_copy['doy_cos'] = np.cos(2 * np.pi * df_copy['day_of_year'] / 365)

    # Year progression
    df_copy['year_progress'] = df_copy['day_of_year'] / 365.0

    # Weekend flag
    df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)

    return df_copy


#  Price Difference Features
def add_price_momentum(df):
    df_copy = df.copy()

    # Price differences
    df_copy['price_diff_1'] = df_copy['Paddy_Price_LKR_per_kg'].diff()
    df_copy['price_diff_3'] = df_copy['Paddy_Price_LKR_per_kg'].diff(3)
    df_copy['price_diff_7'] = df_copy['Paddy_Price_LKR_per_kg'].diff(7)

    # Price momentum (rate of change)
    df_copy['price_momentum_3'] = df_copy['Paddy_Price_LKR_per_kg'].pct_change(3)
    df_copy['price_momentum_7'] = df_copy['Paddy_Price_LKR_per_kg'].pct_change(7)

    # Price volatility
    df_copy['price_volatility_7'] = df_copy['Paddy_Price_LKR_per_kg'].rolling(7).std()

    return df_copy


# Data Validation
def validate_data(df, target_col):
    print("\nData Validation:")
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values in target: {df[target_col].isnull().sum()}")
    print(f"Target statistics:")
    print(f"  Mean: {df[target_col].mean():.2f}")
    print(f"  Std: {df[target_col].std():.2f}")
    print(f"  Min: {df[target_col].min():.2f}")
    print(f"  Max: {df[target_col].max():.2f}")

    # Check for constant values
    if df[target_col].std() < 0.01:
        print("Target variable has very low variance!")

    return df


#  Enhanced LSTM Sequence Creation
def create_enhanced_lstm_sequences(df, feature_cols, target_col, window_size=14):
    X, y = create_lstm_sequences(df, feature_cols, target_col, window_size)

    print(f"LSTM Sequences created:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Target range: {y.min():.2f} to {y.max():.2f}")
    print(f"  Target std: {y.std():.2f}")

    # Check NaN values
    if np.isnan(X).any() or np.isnan(y).any():
        print("NaN values detected in sequences!")

        X = np.nan_to_num(X)
        y = np.nan_to_num(y)

    return X, y


# Main Training Pipeline
def main():
    os.makedirs("models", exist_ok=True)

    # Load & preprocess dataset
    print("Loading dataset...")
    df = load_dataset()

    # Validate original data
    df = validate_data(df, "Paddy_Price_LKR_per_kg")

    df_raw, df_mm, df_std, artifacts = preprocess(df, save_artifacts=True, artifact_dir="models")

    # Enhanced feature engineering for LSTM
    print("\nApplying enhanced feature engineering...")
    df_mm = add_rolling_and_seasonal(df_mm)
    df_mm = add_price_momentum(df_mm)
    df_mm = df_mm.dropna().reset_index(drop=True)

    # LSTM features for price prediction
    exclude = {"Date", "Paddy_Price_LKR_per_kg", "Demand_Tons"}
    feature_cols = [c for c in df_mm.columns if c not in exclude]

    print(f"LSTM Feature columns ({len(feature_cols)}): {feature_cols}")

    # Train Enhanced LSTM Price Model
    print("\nCreating enhanced LSTM sequences for price prediction...")
    window_size = 21  # Increased window size for better context
    X_seq, y_seq = create_enhanced_lstm_sequences(
        df_mm, feature_cols, target_col="Paddy_Price_LKR_per_kg", window_size=window_size
    )

    train_size = int(0.8 * len(X_seq))
    X_train_seq, X_val_seq = X_seq[:train_size], X_seq[train_size:]
    y_train_seq, y_val_seq = y_seq[:train_size], y_seq[train_size:]

    print(f"Training sequences: {X_train_seq.shape}")
    print(f"Validation sequences: {X_val_seq.shape}")

    # Clear session and build enhanced model
    tf.keras.backend.clear_session()
    lstm = build_enhanced_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))

    print("\nEnhanced LSTM Model Architecture:")
    lstm.summary()

    # Enhanced callbacks
    checkpoint_path = os.path.join("models", "best_lstm_price_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss", verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-7)
    ]

    print("\nTraining Enhanced LSTM model...")
    history = lstm.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=200,  # Increased epochs
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
        shuffle=False  # Important for time series
    )

    # Evaluate LSTM model
    train_pred = lstm.predict(X_train_seq, verbose=0)
    val_pred = lstm.predict(X_val_seq, verbose=0)

    train_rmse = np.sqrt(mean_squared_error(y_train_seq, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val_seq, val_pred))
    train_r2 = r2_score(y_train_seq, train_pred)
    val_r2 = r2_score(y_val_seq, val_pred)

    print(f"\nLSTM Model Evaluation:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Val RMSE: {val_rmse:.4f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Val R²: {val_r2:.4f}")

    # Save the final model
    lstm.save(os.path.join("models", "lstm_price_model_final.h5"))
    print("Enhanced LSTM model saved")

    #  Feature Engineering for Demand
    print("\nPreparing features for demand prediction...")

    # Predict
    X_full_seq, _ = create_enhanced_lstm_sequences(
        df_mm, feature_cols, target_col="Paddy_Price_LKR_per_kg", window_size=window_size
    )
    pred_prices = lstm.predict(X_full_seq, verbose=0)

    # Create proper aligned predictions
    pred_prices_full = np.zeros(len(df_std))
    pred_prices_full[window_size:window_size + len(pred_prices)] = pred_prices.flatten()
    pred_prices_full[:window_size] = pred_prices[0]  # Fill initial values

    df_std['Price_LSTM_pred'] = pred_prices_full

    # Enhanced feature engineering for demand
    df_std = add_lag_features(df_std, "Demand_Tons", n_lags=21)
    df_std = add_rolling_and_seasonal(df_std)
    df_std = add_price_momentum(df_std)
    df_std = df_std.dropna().reset_index(drop=True)

    # Encode categorical columns
    cat_cols = [c for c in df_std.columns if df_std[c].dtype == 'object' and c != 'Date']
    for col in cat_cols:
        le = LabelEncoder()
        df_std[col] = le.fit_transform(df_std[col])
        joblib.dump(le, f"models/{col}_encoder.joblib")

    # Prepare features and target
    target_col = "Demand_Tons"
    X_tab = df_std.drop(columns=[target_col, "Date"])
    y_tab = df_std[target_col].values

    print(f"Final feature matrix shape: {X_tab.shape}")

    # Time-based train/test split
    split_idx = int(0.8 * len(X_tab))
    X_train_tab, X_test_tab = X_tab.iloc[:split_idx], X_tab.iloc[split_idx:]
    y_train_tab, y_test_tab = y_tab[:split_idx], y_tab[split_idx:]
    print(f"Tabular data shapes -> Train: {X_train_tab.shape}, Test: {X_test_tab.shape}")

    # Train Enhanced XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )

    print("\nTraining Enhanced XGBoost Demand model...")
    model_xgb.fit(
        X_train_tab, y_train_tab,
        eval_set=[(X_test_tab, y_test_tab)],
        early_stopping_rounds=100,
        verbose=50
    )

    # Enhanced evaluation
    y_pred = model_xgb.predict(X_test_tab)
    r2 = r2_score(y_test_tab, y_pred)
    mae = mean_absolute_error(y_test_tab, y_pred)
    rmse = mean_squared_error(y_test_tab, y_pred, squared=False)
    mape = np.mean(np.abs((y_test_tab - y_pred) / y_test_tab)) * 100

    print("\nEnhanced XGBoost Demand Model Evaluation:")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Feature importance
    importances = model_xgb.feature_importances_
    feat_imp = sorted(zip(X_tab.columns, importances), key=lambda x: x[1], reverse=True)
    print("\n🔝 Top 15 Important Features for Demand:")
    for f, imp in feat_imp[:15]:
        print(f"  {f}: {imp:.4f}")

    # Save models and artifacts
    joblib.dump(model_xgb, "models/xgb_demand_model_best_optimized.joblib")
    joblib.dump(X_tab.columns.tolist(), "models/feature_columns_optimized.joblib")
    joblib.dump(feature_cols, "models/lstm_feature_columns.joblib")

    # Save preprocessing info
    training_info = {
        'window_size': window_size,
        'lstm_features': feature_cols,
        'xgb_features': X_tab.columns.tolist(),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_shape': df.shape,
        'final_features': X_tab.shape[1]
    }
    joblib.dump(training_info, "models/training_info.joblib")

    print("\nAll models and artifacts saved in 'models/' folder.")
    print(f"Models saved:")
    print(f"   - LSTM Price Model: lstm_price_model_final.h5")
    print(f"   - XGBoost Demand Model: xgb_demand_model_best_optimized.joblib")
    print(f"   - Feature columns: feature_columns_optimized.joblib")
    print(f"   - Training info: training_info.joblib")


if __name__ == "__main__":
    main()