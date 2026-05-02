import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "paddy_price_dataset_with_rice_type_stronger_timeseries.xlsx"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)


# =========================================================
# LOAD DATASET
# =========================================================
def load_dataset(path=DATA_PATH):
    df = pd.read_excel(path, sheet_name=0)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Region", "Rice_Type", "Date"]).reset_index(drop=True)
    return df


# =========================================================
# CLEAN DATASET
# =========================================================
def clean_dataset(df):
    df = df.copy()

    # safety cleaning
    df["Rainfall_mm"] = df["Rainfall_mm"].clip(lower=0)
    df["Phosphorus_P"] = df["Phosphorus_P"].clip(lower=0)
    df["Nitrogen_N"] = df["Nitrogen_N"].clip(lower=0)
    df["Potassium_K"] = df["Potassium_K"].clip(lower=0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df = df.dropna(subset=["Date", "Region", "Rice_Type"]).reset_index(drop=True)
    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def add_time_features(df):
    df = df.copy()

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

    # lags
    for col in lag_base_cols:
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            df[f"{col}_lag{lag}"] = df.groupby(group_cols)[col].shift(lag)

    # rolling means
    for col in ["Paddy_Price_LKR_per_kg", "Demand_Tons", "Rainfall_mm", "Temperature_C"]:
        for win in [3, 7, 14, 21, 30]:
            df[f"{col}_roll{win}"] = (
                df.groupby(group_cols)[col]
                .rolling(win)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )

    # price trend features
    df["Price_diff_1"] = df.groupby(group_cols)["Paddy_Price_LKR_per_kg"].diff(1)
    df["Price_diff_3"] = df.groupby(group_cols)["Paddy_Price_LKR_per_kg"].diff(3)
    df["Price_diff_7"] = df.groupby(group_cols)["Paddy_Price_LKR_per_kg"].diff(7)

    df["Price_momentum_7"] = df.groupby(group_cols)["Paddy_Price_LKR_per_kg"].pct_change(7)
    df["Price_momentum_14"] = df.groupby(group_cols)["Paddy_Price_LKR_per_kg"].pct_change(14)

    # demand trend features
    df["Demand_diff_1"] = df.groupby(group_cols)["Demand_Tons"].diff(1)
    df["Demand_momentum_7"] = df.groupby(group_cols)["Demand_Tons"].pct_change(7)

    return df


# =========================================================
# ENCODING
# =========================================================
def encode_categories(df):
    df = df.copy()

    le_region = LabelEncoder()
    le_rice = LabelEncoder()
    le_season = LabelEncoder()

    df["Region_encoded"] = le_region.fit_transform(df["Region"])
    df["Rice_Type_encoded"] = le_rice.fit_transform(df["Rice_Type"])
    df["Season_encoded"] = le_season.fit_transform(df["Season"])

    joblib.dump(le_region, os.path.join(MODEL_DIR, "region_encoder.joblib"))
    joblib.dump(le_rice, os.path.join(MODEL_DIR, "rice_type_encoder.joblib"))
    joblib.dump(le_season, os.path.join(MODEL_DIR, "season_encoder.joblib"))

    return df


# =========================================================
# SEQUENCE CREATION
# =========================================================
def create_sequences(df, feature_cols, target_col, group_cols, window_size=30):
    X_all, y_all = [], []

    grouped = df.groupby(group_cols, sort=False)

    for _, g in grouped:
        g = g.sort_values("Date").reset_index(drop=True)

        if len(g) <= window_size:
            continue

        X_group = g[feature_cols].values
        y_group = g[target_col].values

        for i in range(window_size, len(g)):
            X_all.append(X_group[i - window_size:i])
            y_all.append(y_group[i])

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)
    return X_all, y_all


# =========================================================
# MODEL
# =========================================================
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(96, return_sequences=True, input_shape=input_shape),
        Dropout(0.15),

        LSTM(48, return_sequences=False),
        Dropout(0.10),

        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# =========================================================
# PLOTTING
# =========================================================
def plot_predictions(y_true, y_pred, title, save_path, ylabel):
    plt.figure(figsize=(14, 6))
    plt.plot(y_true[:250], label="Actual")
    plt.plot(y_pred[:250], label="Predicted")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("Cleaning dataset...")
    df = clean_dataset(df)

    print("Creating features...")
    df = add_time_features(df)
    df = add_group_features(df)
    df = encode_categories(df)

    # remove NaNs from lag/rolling features
    df = df.dropna().reset_index(drop=True)

    # =====================================================
    # PRICE MODEL (LSTM)
    # =====================================================
    price_feature_cols = [
        "Region_encoded",
        "Rice_Type_encoded",
        "Season_encoded",
        "Month",
        "DayOfYear",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
        "Rainfall_mm",
        "Temperature_C",
        "Sentiment_Score",
        "News_Sentiment",
        "Paddy_Price_LKR_per_kg_lag1",
        "Paddy_Price_LKR_per_kg_lag2",
        "Paddy_Price_LKR_per_kg_lag3",
        "Paddy_Price_LKR_per_kg_lag7",
        "Paddy_Price_LKR_per_kg_lag14",
        "Paddy_Price_LKR_per_kg_lag21",
        "Paddy_Price_LKR_per_kg_lag30",
        "Paddy_Price_LKR_per_kg_roll3",
        "Paddy_Price_LKR_per_kg_roll7",
        "Paddy_Price_LKR_per_kg_roll14",
        "Paddy_Price_LKR_per_kg_roll21",
        "Paddy_Price_LKR_per_kg_roll30",
        "Price_diff_1",
        "Price_diff_3",
        "Price_diff_7",
        "Price_momentum_7",
        "Price_momentum_14"
    ]

    target_price = "Paddy_Price_LKR_per_kg"
    window_size = 30

    # chronological split
    split_date = df["Date"].quantile(0.80)
    df_train = df[df["Date"] <= split_date].copy()
    df_val = df[df["Date"] > split_date].copy()

    print(f"Train rows: {len(df_train)}")
    print(f"Val rows: {len(df_val)}")
    print(f"Split date: {split_date}")

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()

    df_train_scaled[price_feature_cols] = feature_scaler.fit_transform(df_train_scaled[price_feature_cols])
    df_val_scaled[price_feature_cols] = feature_scaler.transform(df_val_scaled[price_feature_cols])

    df_train_scaled[[target_price]] = target_scaler.fit_transform(df_train_scaled[[target_price]])
    df_val_scaled[[target_price]] = target_scaler.transform(df_val_scaled[[target_price]])

    joblib.dump(feature_scaler, os.path.join(MODEL_DIR, "lstm_feature_scaler.joblib"))
    joblib.dump(target_scaler, os.path.join(MODEL_DIR, "lstm_target_scaler.joblib"))
    joblib.dump(price_feature_cols, os.path.join(MODEL_DIR, "lstm_feature_columns.joblib"))

    X_train, y_train = create_sequences(
        df_train_scaled,
        feature_cols=price_feature_cols,
        target_col=target_price,
        group_cols=["Region", "Rice_Type"],
        window_size=window_size
    )

    X_val, y_val = create_sequences(
        df_val_scaled,
        feature_cols=price_feature_cols,
        target_col=target_price,
        group_cols=["Region", "Rice_Type"],
        window_size=window_size
    )

    print("LSTM train shape:", X_train.shape)
    print("LSTM val shape:", X_val.shape)

    tf.keras.backend.clear_session()
    lstm = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, "best_lstm_price_model.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\nTraining LSTM price model...")
    lstm.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )

    y_train_pred_scaled = lstm.predict(X_train, verbose=0)
    y_val_pred_scaled = lstm.predict(X_val, verbose=0)

    y_train_true = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_val_true = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_train_pred = target_scaler.inverse_transform(y_train_pred_scaled).flatten()
    y_val_pred = target_scaler.inverse_transform(y_val_pred_scaled).flatten()

    print("\n===== LSTM PRICE MODEL =====")
    print("Train RMSE:", np.sqrt(mean_squared_error(y_train_true, y_train_pred)))
    print("Val RMSE:", np.sqrt(mean_squared_error(y_val_true, y_val_pred)))
    print("Train MAE :", mean_absolute_error(y_train_true, y_train_pred))
    print("Val MAE   :", mean_absolute_error(y_val_true, y_val_pred))
    print("Train R2  :", r2_score(y_train_true, y_train_pred))
    print("Val R2    :", r2_score(y_val_true, y_val_pred))

    lstm.save(os.path.join(MODEL_DIR, "lstm_price_model_final.h5"))

    plot_predictions(
        y_val_true,
        y_val_pred,
        "LSTM Price Prediction - Actual vs Predicted",
        os.path.join(MODEL_DIR, "lstm_price_prediction_curve.png"),
        ylabel="Price"
    )

    # =====================================================
    # PRICE PREDICTIONS FOR DEMAND MODEL
    # =====================================================
    df_all_scaled = df.copy()
    df_all_scaled[price_feature_cols] = feature_scaler.transform(df_all_scaled[price_feature_cols])
    df_all_scaled[[target_price]] = target_scaler.transform(df_all_scaled[[target_price]])

    X_full, _ = create_sequences(
        df_all_scaled,
        feature_cols=price_feature_cols,
        target_col=target_price,
        group_cols=["Region", "Rice_Type"],
        window_size=window_size
    )

    full_pred_scaled = lstm.predict(X_full, verbose=0)
    full_pred = target_scaler.inverse_transform(full_pred_scaled).flatten()

    df_demand = df.copy().reset_index(drop=True)
    df_demand["Price_LSTM_pred"] = np.nan

    idx_positions = []
    for _, g in df_demand.groupby(["Region", "Rice_Type"], sort=False):
        g = g.sort_values("Date")
        idxs = g.index.tolist()
        if len(idxs) > window_size:
            idx_positions.extend(idxs[window_size:])

    df_demand.loc[idx_positions, "Price_LSTM_pred"] = full_pred

    df_demand["Price_LSTM_pred"] = (
        df_demand.groupby(["Region", "Rice_Type"])["Price_LSTM_pred"]
        .transform(lambda s: s.bfill().ffill())
    )

    # =====================================================
    # DEMAND MODEL (XGBOOST)
    # =====================================================
    demand_feature_cols = [
        "Region_encoded",
        "Rice_Type_encoded",
        "Season_encoded",
        "Month",
        "DayOfYear",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
        "Rainfall_mm",
        "Temperature_C",
        "Sentiment_Score",
        "News_Sentiment",
        "Nitrogen_N",
        "Phosphorus_P",
        "Potassium_K",
        "Price_LSTM_pred",
        "Paddy_Price_LKR_per_kg_lag1",
        "Paddy_Price_LKR_per_kg_lag2",
        "Paddy_Price_LKR_per_kg_lag3",
        "Paddy_Price_LKR_per_kg_lag7",
        "Paddy_Price_LKR_per_kg_lag14",
        "Paddy_Price_LKR_per_kg_roll3",
        "Paddy_Price_LKR_per_kg_roll7",
        "Paddy_Price_LKR_per_kg_roll14",
        "Demand_Tons_lag1",
        "Demand_Tons_lag2",
        "Demand_Tons_lag3",
        "Demand_Tons_lag7",
        "Demand_Tons_lag14",
        "Demand_Tons_roll3",
        "Demand_Tons_roll7",
        "Demand_Tons_roll14",
        "Price_diff_1",
        "Price_diff_3",
        "Price_diff_7",
        "Price_momentum_7",
        "Demand_diff_1",
        "Demand_momentum_7"
    ]

    target_demand = "Demand_Tons"

    split_date_demand = df_demand["Date"].quantile(0.80)
    train_mask = df_demand["Date"] <= split_date_demand
    test_mask = df_demand["Date"] > split_date_demand

    X_train_tab = df_demand.loc[train_mask, demand_feature_cols].copy()
    y_train_tab = df_demand.loc[train_mask, target_demand].values

    X_test_tab = df_demand.loc[test_mask, demand_feature_cols].copy()
    y_test_tab = df_demand.loc[test_mask, target_demand].values

    model_xgb = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=2,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.03,
        reg_lambda=0.20,
        objective="reg:squarederror",
        random_state=42
    )

    print("\nTraining XGBoost demand model...")
    model_xgb.fit(
        X_train_tab,
        y_train_tab,
        eval_set=[(X_test_tab, y_test_tab)],
        verbose=100
    )

    y_pred = model_xgb.predict(X_test_tab)

    print("\n===== XGBOOST DEMAND MODEL =====")
    print("R2   :", r2_score(y_test_tab, y_pred))
    print("MAE  :", mean_absolute_error(y_test_tab, y_pred))
    print("RMSE :", np.sqrt(mean_squared_error(y_test_tab, y_pred)))

    plot_predictions(
        y_test_tab,
        y_pred,
        "XGBoost Demand Prediction - Actual vs Predicted",
        os.path.join(MODEL_DIR, "xgb_demand_prediction_curve.png"),
        ylabel="Demand"
    )

    # =====================================================
    # SAVE FILES
    # =====================================================
    joblib.dump(model_xgb, os.path.join(MODEL_DIR, "xgb_demand_model.joblib"))
    joblib.dump(demand_feature_cols, os.path.join(MODEL_DIR, "xgb_feature_columns.joblib"))

    training_info = {
        "window_size": window_size,
        "lstm_features": price_feature_cols,
        "xgb_features": demand_feature_cols,
        "data_shape": df.shape,
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(training_info, os.path.join(MODEL_DIR, "training_info.joblib"))

    print("\nSaved files:")
    print("- models/lstm_price_model_final.h5")
    print("- models/best_lstm_price_model.keras")
    print("- models/xgb_demand_model.joblib")
    print("- models/lstm_price_prediction_curve.png")
    print("- models/xgb_demand_prediction_curve.png")
    print("- models/training_info.joblib")


if __name__ == "__main__":
    main()