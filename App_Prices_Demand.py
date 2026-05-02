from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, make_response
import os
import sys
import io
import re
import time
import base64
import joblib
import serial
import threading
import traceback
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRICES_DEMAND_DIR = os.path.join(BASE_DIR, 'Prices_Demand')

sys.path.insert(0, PRICES_DEMAND_DIR)

try:
    from load import load_dataset
    from pre_process import preprocess
    from utils import (
        add_lag_features, add_rolling_and_seasonal, add_price_momentum,
        calculate_historical_trend, update_time_features
    )
    from run_prediction import predict_price_with_trend, predict_demand_with_trend
    print("Successfully imported all modules from Prices_Demand")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Looking in: {PRICES_DEMAND_DIR}")
    print(f"Files in directory: {os.listdir(PRICES_DEMAND_DIR)}")
    sys.exit(1)

app = Flask(
    __name__,
    template_folder=os.path.join(PRICES_DEMAND_DIR, 'templates'),
    static_folder=os.path.join(PRICES_DEMAND_DIR, 'static')
)

app.secret_key = 'paddy_prediction_secret_key_2024'
app.config['UPLOAD_FOLDER'] = os.path.join(PRICES_DEMAND_DIR, 'static', 'uploads')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(PRICES_DEMAND_DIR, 'static', 'images'), exist_ok=True)

# Global variables
models = None
historical_data = None
df_raw = None
df_mm = None
df_std = None
latest_prediction_result = None


# =========================
# LIVE SENSOR MONITOR
# =========================
class LiveSensorMonitor:
    def __init__(self, port="COM7", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        self.latest = {
            "connected": False,
            "port": self.port,
            "baudrate": self.baudrate,
            "last_raw": "",
            "last_decoded": "",
            "last_update": None,
            "values": {
                "nitrogen": None,
                "phosphorus": None,
                "potassium": None
            }
        }

        # temporary buffer for multi-line packets
        self.partial_values = {}
        self.disconnect_timeout_seconds = 3.0
        self.last_signal_at = None

    def _connection_alive(self):
        return bool(
            self.running
            and self.ser
            and self.ser.is_open
            and self._has_recent_signal()
        )

    def _has_recent_signal(self):
        return bool(
            self.last_signal_at is not None
            and (time.monotonic() - self.last_signal_at) <= self.disconnect_timeout_seconds
        )

    def connect(self):
        try:
            print(f"[SENSOR] Trying to connect to {self.port} at {self.baudrate}...")

            if self.ser and self.ser.is_open:
                self.ser.close()

            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            self.last_signal_at = None

            with self.lock:
                self.latest["connected"] = True
                self.latest["port"] = self.port
                self.latest["baudrate"] = self.baudrate

            print(f"[SENSOR] Connected to {self.port} at {self.baudrate}")
            return True

        except Exception as e:
            with self.lock:
                self.latest["connected"] = False
            print(f"[SENSOR] Could not connect to {self.port}: {e}")
            traceback.print_exc()
            return False

    def start(self):
        if self.thread and self.thread.is_alive():
            return

        if not self.running:
            ok = self.connect()
            if not ok:
                return

        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()
        print("[SENSOR] Live sensor thread started")

    def stop(self):
        self.running = False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

        self.last_signal_at = None
        with self.lock:
            self.latest["connected"] = False

    def parse_line(self, text):
        """
        Supports:
        1) Single-line compact:
           N:520,P:7,K:4
        2) Single-line named:
           nitrogen=520,phosphorus=7,potassium=4
        3) Multi-line style:
           Nitrogen (N): 520
           Phosphorus (P): 7
           Potassium (K): 4
        """
        if not text:
            return None

        line = text.strip().lower()

        # Ignore decorative lines
        if "npk values" in line or "----" in line:
            return None

        extracted = {}

        patterns = {
            "nitrogen": [
                r"\bn\s*[:=]\s*(-?\d+(?:\.\d+)?)",
                r"\bnitrogen\s*\(n\)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
                r"\bnitrogen\s*[:=]\s*(-?\d+(?:\.\d+)?)"
            ],
            "phosphorus": [
                r"\bp\s*[:=]\s*(-?\d+(?:\.\d+)?)",
                r"\bphosphorus\s*\(p\)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
                r"\bphosphorus\s*[:=]\s*(-?\d+(?:\.\d+)?)"
            ],
            "potassium": [
                r"\bk\s*[:=]\s*(-?\d+(?:\.\d+)?)",
                r"\bpotassium\s*\(k\)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
                r"\bpotassium\s*[:=]\s*(-?\d+(?:\.\d+)?)"
            ]
        }

        for key, regex_list in patterns.items():
            for pattern in regex_list:
                match = re.search(pattern, line)
                if match:
                    extracted[key] = float(match.group(1))
                    break

        return extracted if extracted else None

    def commit_partial_values(self):
        """Move parsed multi-line values into latest sensor values."""
        if not self.partial_values:
            return

        with self.lock:
            for key, value in self.partial_values.items():
                self.latest["values"][key] = value
            self.latest["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.partial_values = {}

    def read_loop(self):
        while self.running:
            try:
                raw = self.ser.readline()

                if raw:
                    try:
                        text = raw.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        text = raw.decode(errors="ignore").strip()

                    print(f"[SENSOR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -> {text}")

                    parsed = self.parse_line(text)
                    self.last_signal_at = time.monotonic()

                    with self.lock:
                        self.latest["last_raw"] = str(raw)
                        self.latest["last_decoded"] = text
                        self.latest["connected"] = True

                    # If a line contained sensor values, store them temporarily
                    if parsed:
                        self.partial_values.update(parsed)

                    # If all NPK arrived, commit immediately
                    if all(k in self.partial_values for k in ["nitrogen", "phosphorus", "potassium"]):
                        self.commit_partial_values()

                    # If separator line arrived, also commit what we have
                    if "----" in text or "npk values" in text.lower():
                        self.commit_partial_values()
                else:
                    with self.lock:
                        self.latest["connected"] = self._connection_alive()

            except Exception:
                print("[SENSOR] Serial read error:")
                traceback.print_exc()
                with self.lock:
                    self.latest["connected"] = False
                time.sleep(1)

    def get_latest(self):
        with self.lock:
            payload = {
                "connected": self.latest["connected"],
                "port": self.latest["port"],
                "baudrate": self.latest["baudrate"],
                "last_raw": self.latest["last_raw"],
                "last_decoded": self.latest["last_decoded"],
                "last_update": self.latest["last_update"],
                "values": dict(self.latest["values"])
            }

        payload["connected"] = self._connection_alive()

        payload["values"]["nitrogen"] = payload["values"]["nitrogen"] if payload["values"]["nitrogen"] is not None else 520.0
        payload["values"]["phosphorus"] = payload["values"]["phosphorus"] if payload["values"]["phosphorus"] is not None else 10.0
        payload["values"]["potassium"] = payload["values"]["potassium"] if payload["values"]["potassium"] is not None else 10.0

        return payload


sensor_monitor = LiveSensorMonitor(port="COM7", baudrate=115200)


# =========================
# MODEL / DATA LOADING
# =========================
def load_models():
    models_dict = {}
    models_path = os.path.join(PRICES_DEMAND_DIR, 'models')

    try:
        if not os.path.exists(models_path):
            print(f"Models directory not found at {models_path}")
            return None

        lstm_path = os.path.join(models_path, 'lstm_price_model_final.h5')
        if os.path.exists(lstm_path):
            models_dict['lstm'] = load_model(lstm_path)
            print("LSTM price model loaded")
        else:
            print(f"LSTM model not found at {lstm_path}")
            return None

        xgb_path = os.path.join(models_path, 'xgb_demand_model_best_optimized.joblib')
        if os.path.exists(xgb_path):
            models_dict['xgb'] = joblib.load(xgb_path)
            print("XGBoost demand model loaded")
        else:
            print(f"XGBoost model not found at {xgb_path}")
            return None

        xgb_features_path = os.path.join(models_path, 'feature_columns_optimized.joblib')
        if os.path.exists(xgb_features_path):
            models_dict['xgb_features'] = joblib.load(xgb_features_path)
            print(f"Loaded {len(models_dict['xgb_features'])} demand features")
        else:
            print("Feature columns not found")
            return None

        lstm_features_path = os.path.join(models_path, 'lstm_feature_columns.joblib')
        if os.path.exists(lstm_features_path):
            models_dict['lstm_features'] = joblib.load(lstm_features_path)
            print(f"Loaded {len(models_dict['lstm_features'])} LSTM features")
        else:
            print("LSTM feature columns not found")
            return None

        training_info_path = os.path.join(models_path, 'training_info.joblib')
        if os.path.exists(training_info_path):
            training_info = joblib.load(training_info_path)
            models_dict['window_size'] = training_info.get('window_size', 21)
            print(f"Training info loaded (window size: {models_dict['window_size']})")
        else:
            models_dict['window_size'] = 21
            print("Training info not found, using default window size: 21")

        return models_dict

    except Exception as e:
        print(f"Error loading models: {e}")
        return None


def load_and_preprocess_data():
    try:
        current_dir = os.getcwd()

        os.chdir(PRICES_DEMAND_DIR)
        print(f"Changed to directory: {os.getcwd()}")

        df = load_dataset()
        print(f"Loaded dataset with {len(df)} rows")

        df_raw_local, df_mm_local, df_std_local, _ = preprocess(df, save_artifacts=False)

        os.chdir(current_dir)
        print(f"Changed back to: {os.getcwd()}")

        df_mm_local = add_rolling_and_seasonal(df_mm_local)
        df_mm_local = add_price_momentum(df_mm_local)
        df_mm_local = df_mm_local.dropna().reset_index(drop=True)

        print(
            f"Preprocessed data shapes: Raw: {df_raw_local.shape}, "
            f"MM: {df_mm_local.shape}, STD: {df_std_local.shape}"
        )

        return df_raw_local, df_mm_local, df_std_local

    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return None, None, None


# =========================
# HISTORY / PLOTTING
# =========================
def build_clean_historical_series(df, value_col, history_days=14, smooth_window=3, region=None):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=['Date', value_col, f'{value_col}_smooth'])

    hist_df = df.copy()
    hist_df['Date'] = pd.to_datetime(hist_df['Date'])

    if region is not None and 'Region' in hist_df.columns:
        hist_df = hist_df[hist_df['Region'] == region].copy()

    hist_df = (
        hist_df.groupby('Date', as_index=False)[value_col]
        .mean()
        .sort_values('Date')
        .reset_index(drop=True)
    )

    hist_df = hist_df.tail(history_days).copy()

    if smooth_window and smooth_window > 1:
        hist_df[f'{value_col}_smooth'] = (
            hist_df[value_col]
            .rolling(window=smooth_window, min_periods=1)
            .mean()
        )
    else:
        hist_df[f'{value_col}_smooth'] = hist_df[value_col]

    return hist_df


def get_historical_data(df):
    if df is None or len(df) == 0:
        return {}

    try:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])

        price_hist = build_clean_historical_series(
            df, 'Paddy_Price_LKR_per_kg', history_days=14, smooth_window=3
        )
        demand_hist = build_clean_historical_series(
            df, 'Demand_Tons', history_days=14, smooth_window=3
        )

        daily_df = (
            df.groupby('Date', as_index=False)[['Paddy_Price_LKR_per_kg', 'Demand_Tons']]
            .mean()
            .sort_values('Date')
            .reset_index(drop=True)
        )

        return {
            'date_range': {
                'start': daily_df['Date'].min().strftime('%Y-%m-%d'),
                'end': daily_df['Date'].max().strftime('%Y-%m-%d'),
                'total_days': int(len(daily_df))
            },
            'price_stats': {
                'mean': float(daily_df['Paddy_Price_LKR_per_kg'].mean()),
                'min': float(daily_df['Paddy_Price_LKR_per_kg'].min()),
                'max': float(daily_df['Paddy_Price_LKR_per_kg'].max()),
                'std': float(daily_df['Paddy_Price_LKR_per_kg'].std())
            },
            'demand_stats': {
                'mean': float(daily_df['Demand_Tons'].mean()),
                'min': float(daily_df['Demand_Tons'].min()),
                'max': float(daily_df['Demand_Tons'].max()),
                'std': float(daily_df['Demand_Tons'].std())
            },
            'recent_dates': price_hist['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'recent_prices': price_hist['Paddy_Price_LKR_per_kg_smooth'].tolist(),
            'recent_demand': demand_hist['Demand_Tons_smooth'].tolist(),
            'regions': df['Region'].unique().tolist() if 'Region' in df.columns else []
        }

    except Exception as e:
        print(f"Error getting historical data: {e}")
        return {}


def prepare_continuous_series(df_raw, prediction_dates, predictions, value_col, history_days=14, smooth_window=3):
    hist_df = build_clean_historical_series(
        df_raw,
        value_col=value_col,
        history_days=history_days,
        smooth_window=smooth_window
    )

    hist_dates = hist_df['Date'].reset_index(drop=True)
    hist_values = hist_df[f'{value_col}_smooth'].values.astype(float)

    pred_values = np.array(predictions, dtype=float).copy()

    if len(hist_values) > 0 and len(pred_values) > 0:
        offset = hist_values[-1] - pred_values[0]
        pred_values = pred_values + offset

    if len(hist_dates) > 0:
        plot_pred_dates = pd.date_range(
            start=hist_dates.iloc[-1] + pd.Timedelta(days=1),
            periods=len(pred_values),
            freq='D'
        )
    else:
        plot_pred_dates = pd.to_datetime(prediction_dates)

    return hist_dates, hist_values, plot_pred_dates, pred_values


def generate_price_plot(df_raw, prediction_dates, price_predictions):
    plt.figure(figsize=(12, 5))

    hist_dates, hist_prices, plot_pred_dates, plot_pred_prices = prepare_continuous_series(
        df_raw, prediction_dates, price_predictions,
        'Paddy_Price_LKR_per_kg', history_days=14, smooth_window=3
    )

    plt.plot(hist_dates, hist_prices, 'b-', label='Historical Price', linewidth=2, alpha=0.8)

    if len(hist_dates) > 0 and len(plot_pred_dates) > 0:
        plt.plot(
            [hist_dates.iloc[-1], plot_pred_dates[0]],
            [hist_prices[-1], plot_pred_prices[0]],
            color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='Transition'
        )

    plt.plot(plot_pred_dates, plot_pred_prices, 'ro-', label='Predicted Price', linewidth=2, markersize=5)

    if len(hist_dates) > 0:
        plt.axvline(x=hist_dates.iloc[-1], color='gray', linestyle='--', alpha=0.6, label='Prediction Start')

    plt.title('Paddy Price Prediction', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Price (LKR/kg)', fontsize=11)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


def generate_demand_plot(df_raw, prediction_dates, demand_predictions):
    plt.figure(figsize=(12, 5))

    hist_dates, hist_demand, plot_pred_dates, plot_pred_demand = prepare_continuous_series(
        df_raw, prediction_dates, demand_predictions,
        'Demand_Tons', history_days=14, smooth_window=3
    )

    plt.plot(hist_dates, hist_demand, 'g-', label='Historical Demand', linewidth=2, alpha=0.8)

    if len(hist_dates) > 0 and len(plot_pred_dates) > 0:
        plt.plot(
            [hist_dates.iloc[-1], plot_pred_dates[0]],
            [hist_demand[-1], plot_pred_demand[0]],
            color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='Transition'
        )

    plt.plot(plot_pred_dates, plot_pred_demand, 'mo-', label='Predicted Demand', linewidth=2, markersize=5)

    if len(hist_dates) > 0:
        plt.axvline(x=hist_dates.iloc[-1], color='gray', linestyle='--', alpha=0.6, label='Prediction Start')

    plt.title('Demand Prediction', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Demand (Tons)', fontsize=11)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


def generate_combined_plot(df_raw, prediction_dates, price_predictions, demand_predictions):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    price_hist_df = build_clean_historical_series(
        df_raw, 'Paddy_Price_LKR_per_kg', history_days=14, smooth_window=3
    )
    demand_hist_df = build_clean_historical_series(
        df_raw, 'Demand_Tons', history_days=14, smooth_window=3
    )

    hist_price_dates = price_hist_df['Date'].reset_index(drop=True)
    hist_prices = price_hist_df['Paddy_Price_LKR_per_kg_smooth'].values.astype(float)

    hist_demand_dates = demand_hist_df['Date'].reset_index(drop=True)
    hist_demands = demand_hist_df['Demand_Tons_smooth'].values.astype(float)

    pred_prices = np.array(price_predictions, dtype=float).copy()
    pred_demands = np.array(demand_predictions, dtype=float).copy()

    if len(hist_price_dates) > 0:
        plot_pred_dates = pd.date_range(
            start=hist_price_dates.iloc[-1] + pd.Timedelta(days=1),
            periods=len(pred_prices),
            freq='D'
        )
    else:
        plot_pred_dates = pd.to_datetime(prediction_dates)

    if len(hist_prices) > 0 and len(pred_prices) > 0:
        pred_prices = pred_prices + (hist_prices[-1] - pred_prices[0])

    if len(hist_demands) > 0 and len(pred_demands) > 0:
        pred_demands = pred_demands + (hist_demands[-1] - pred_demands[0])

    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Price (LKR/kg)', color='red', fontsize=11)
    line1 = ax1.plot(hist_price_dates, hist_prices, 'r-', linewidth=2, alpha=0.6, label='Historical Price')
    line2 = ax1.plot(plot_pred_dates, pred_prices, 'ro-', linewidth=2, markersize=5, label='Predicted Price')

    if len(hist_price_dates) > 0 and len(plot_pred_dates) > 0:
        ax1.plot(
            [hist_price_dates.iloc[-1], plot_pred_dates[0]],
            [hist_prices[-1], pred_prices[0]],
            color='orange', linestyle='--', linewidth=1.5, alpha=0.8
        )

    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Demand (Tons)', color='blue', fontsize=11)
    line3 = ax2.plot(hist_demand_dates, hist_demands, 'b-', linewidth=2, alpha=0.6, label='Historical Demand')
    line4 = ax2.plot(plot_pred_dates, pred_demands, 'bs-', linewidth=2, markersize=5, label='Predicted Demand')

    if len(hist_demand_dates) > 0 and len(plot_pred_dates) > 0:
        ax2.plot(
            [hist_demand_dates.iloc[-1], plot_pred_dates[0]],
            [hist_demands[-1], pred_demands[0]],
            color='orange', linestyle='--', linewidth=1.5, alpha=0.8
        )

    ax2.tick_params(axis='y', labelcolor='blue')

    if len(hist_price_dates) > 0:
        ax1.axvline(x=hist_price_dates.iloc[-1], color='gray', linestyle='--', alpha=0.6, label='Prediction Start')

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)

    plt.title('Price vs Demand Prediction Trend', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


# =========================
# APP INIT
# =========================
def initialize_app():
    global models, historical_data, df_raw, df_mm, df_std

    print("\n" + "=" * 60)
    print("🌾 PADDY PRICE & DEMAND PREDICTION SYSTEM")
    print("=" * 60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Prices_Demand directory: {PRICES_DEMAND_DIR}")
    print(f"Templates directory: {app.template_folder}")
    print(f"Static directory: {app.static_folder}")

    print("\nLoading models...")
    models = load_models()

    print("\nLoading historical data...")
    df_raw_loaded, df_mm_loaded, df_std_loaded = load_and_preprocess_data()
    df_raw = df_raw_loaded
    df_mm = df_mm_loaded
    df_std = df_std_loaded

    if df_raw is not None:
        historical_data = get_historical_data(df_raw)
        print(
            f"Loaded {len(df_raw)} records from "
            f"{historical_data.get('date_range', {}).get('start', 'N/A')} "
            f"to {historical_data.get('date_range', {}).get('end', 'N/A')}"
        )
    else:
        historical_data = {}
        print("Could not load historical data")

    print("\nStarting live sensor monitor...")
    sensor_monitor.start()

    print("\n" + "=" * 60)
    print("Application initialized successfully!")
    print("=" * 60)


def should_initialize():
    return (not app.debug) or (os.environ.get("WERKZEUG_RUN_MAIN") == "true")


# =========================
# ROUTES
# =========================
@app.context_processor
def utility_processor():
    return {
        'now': datetime.now(),
        'app_name': 'Paddy Price & Demand Predictor',
        'models_loaded': models is not None
    }


@app.route('/')
def index():
    return render_template(
        'index.html',
        historical_data=historical_data,
        models_loaded=models is not None
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            start_date = request.form.get('start_date')
            n_days = int(request.form.get('n_days', 7))

            if not start_date:
                start_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

            start_date = pd.to_datetime(start_date)
            end_date = start_date + timedelta(days=n_days - 1)

            if models is None:
                flash('Models not loaded. Please train models first.', 'error')
                return redirect(url_for('predict'))

            price_predictions, prediction_dates = predict_price_with_trend(
                df_mm, models['lstm'], models['lstm_features'],
                start_date, n_steps=n_days, window_size=models['window_size']
            )

            demand_predictions = predict_demand_with_trend(
                df_std, models['xgb'], models['xgb_features'],
                price_predictions, prediction_dates
            )

            price_plot = generate_price_plot(df_raw, prediction_dates, price_predictions)
            demand_plot = generate_demand_plot(df_raw, prediction_dates, demand_predictions)
            combined_plot = generate_combined_plot(df_raw, prediction_dates, price_predictions, demand_predictions)

            results = {
                'dates': [d.strftime('%Y-%m-%d') for d in prediction_dates],
                'prices': [float(p) for p in price_predictions],
                'demands': [float(d) for d in demand_predictions],
                'avg_price': float(np.mean(price_predictions)),
                'avg_demand': float(np.mean(demand_predictions)),
                'price_range': [float(min(price_predictions)), float(max(price_predictions))],
                'demand_range': [float(min(demand_predictions)), float(max(demand_predictions))],
                'price_trend': 'Increasing' if price_predictions[-1] > price_predictions[0] else 'Decreasing',
                'demand_trend': 'Increasing' if demand_predictions[-1] > demand_predictions[0] else 'Decreasing',
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
            global latest_prediction_result
            latest_prediction_result = {
                'results': results,
                'prediction_type': 'Standard',
                'sensor_data': None,
                'combined_plot': combined_plot
            }

            return render_template(
                'results.html',
                results=results,
                price_plot=price_plot,
                demand_plot=demand_plot,
                combined_plot=combined_plot,
                prediction_type='Standard'
            )

        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('predict'))

    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    return render_template(
        'predict.html',
        today=datetime.now().strftime('%Y-%m-%d'),
        tomorrow=tomorrow
    )


@app.route('/sensor-predict', methods=['GET', 'POST'])
def sensor_predict():
    latest_sensor = sensor_monitor.get_latest()

    if request.method == 'POST':
        try:
            start_date = request.form.get('start_date')
            n_days = int(request.form.get('n_days', 7))

            nitrogen = float(request.form.get('nitrogen', latest_sensor["values"]["nitrogen"]))
            phosphorus = float(request.form.get('phosphorus', latest_sensor["values"]["phosphorus"]))
            potassium = float(request.form.get('potassium', latest_sensor["values"]["potassium"]))

            if not start_date:
                start_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

            start_date = pd.to_datetime(start_date)
            end_date = start_date + timedelta(days=n_days - 1)

            if models is None:
                flash('Models not loaded. Please train models first.', 'error')
                return redirect(url_for('sensor_predict'))

            df_mm_sensor = df_mm.copy()
            df_mm_sensor.loc[df_mm_sensor.index[-1], "Nitrogen_N"] = nitrogen
            df_mm_sensor.loc[df_mm_sensor.index[-1], "Phosphorus_P"] = phosphorus
            df_mm_sensor.loc[df_mm_sensor.index[-1], "Potassium_K"] = potassium

            price_predictions, prediction_dates = predict_price_with_trend(
                df_mm_sensor, models['lstm'], models['lstm_features'],
                start_date, n_steps=n_days, window_size=models['window_size']
            )

            demand_predictions = predict_demand_with_trend(
                df_std, models['xgb'], models['xgb_features'],
                price_predictions, prediction_dates
            )

            price_plot = generate_price_plot(df_raw, prediction_dates, price_predictions)
            demand_plot = generate_demand_plot(df_raw, prediction_dates, demand_predictions)
            combined_plot = generate_combined_plot(df_raw, prediction_dates, price_predictions, demand_predictions)

            results = {
                'dates': [d.strftime('%Y-%m-%d') for d in prediction_dates],
                'prices': [float(p) for p in price_predictions],
                'demands': [float(d) for d in demand_predictions],
                'avg_price': float(np.mean(price_predictions)),
                'avg_demand': float(np.mean(demand_predictions)),
                'price_range': [float(min(price_predictions)), float(max(price_predictions))],
                'demand_range': [float(min(demand_predictions)), float(max(demand_predictions))],
                'price_trend': 'Increasing' if price_predictions[-1] > price_predictions[0] else 'Decreasing',
                'demand_trend': 'Increasing' if demand_predictions[-1] > demand_predictions[0] else 'Decreasing',
                'sensor_data': {
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium,
                    'live_connected': latest_sensor["connected"],
                    'live_last_update': latest_sensor["last_update"]
                },
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
            global latest_prediction_result
            latest_prediction_result = {
                'results': results,
                'prediction_type': 'Sensor-Based',
                'sensor_data': results.get('sensor_data'),
                'combined_plot': combined_plot
            }

            return render_template(
                'results.html',
                results=results,
                price_plot=price_plot,
                demand_plot=demand_plot,
                combined_plot=combined_plot,
                prediction_type='Sensor-Based'
            )

        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('sensor_predict'))

    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    return render_template(
        'sensor_predict.html',
        today=datetime.now().strftime('%Y-%m-%d'),
        tomorrow=tomorrow,
        latest_sensor=latest_sensor
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        start_date = pd.to_datetime(
            data.get('start_date', (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))
        )
        n_days = int(data.get('n_days', 7))

        if models is None:
            return jsonify({'success': False, 'error': 'Models not loaded'}), 500

        price_predictions, prediction_dates = predict_price_with_trend(
            df_mm, models['lstm'], models['lstm_features'],
            start_date, n_steps=n_days, window_size=models['window_size']
        )

        demand_predictions = predict_demand_with_trend(
            df_std, models['xgb'], models['xgb_features'],
            price_predictions, prediction_dates
        )

        response = {
            'success': True,
            'predictions': [
                {
                    'date': d.strftime('%Y-%m-%d'),
                    'price': float(p),
                    'demand': float(dm)
                }
                for d, p, dm in zip(prediction_dates, price_predictions, demand_predictions)
            ],
            'summary': {
                'avg_price': float(np.mean(price_predictions)),
                'avg_demand': float(np.mean(demand_predictions)),
                'price_range': [float(min(price_predictions)), float(max(price_predictions))],
                'demand_range': [float(min(demand_predictions)), float(max(demand_predictions))]
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/download-report')
def download_report():
    global latest_prediction_result

    if not latest_prediction_result:
        flash("No prediction report available. Please generate a prediction first.", "warning")
        return redirect(url_for('predict'))

    results = latest_prediction_result['results']
    prediction_type = latest_prediction_result['prediction_type']
    sensor_data = latest_prediction_result['sensor_data']

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50

    # Title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Paddy Price & Demand Prediction Report")
    y -= 25

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Prediction Type: {prediction_type}")
    y -= 18
    pdf.drawString(50, y, f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 18
    pdf.drawString(50, y, f"Prediction Period: {results['start_date']} to {results['end_date']}")
    y -= 30

    # Summary
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Summary")
    y -= 20

    pdf.setFont("Helvetica", 11)
    pdf.drawString(60, y, f"Average Price: Rs. {results['avg_price']:.2f}")
    y -= 18
    pdf.drawString(60, y, f"Average Demand: {results['avg_demand']:.0f} Tons")
    y -= 18
    pdf.drawString(60, y, f"Price Trend: {results['price_trend']}")
    y -= 18
    pdf.drawString(60, y, f"Demand Trend: {results['demand_trend']}")
    y -= 18
    pdf.drawString(60, y, f"Price Range: {results['price_range'][0]:.2f} - {results['price_range'][1]:.2f} LKR/kg")
    y -= 18
    pdf.drawString(60, y, f"Demand Range: {results['demand_range'][0]:.0f} - {results['demand_range'][1]:.0f} Tons")
    y -= 30

    combined_plot = latest_prediction_result.get('combined_plot')

    if combined_plot:
        y -= 10
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, y, "Price vs Demand Trend Graph")
        y -= 20

        # Decode base64 image
        image_data = base64.b64decode(combined_plot)
        image_stream = io.BytesIO(image_data)

        img = ImageReader(image_stream)

        # Draw image on PDF
        pdf.drawImage(img, 50, y - 250, width=500, height=250)

        y -= 270

    # Sensor data if available
    if sensor_data:
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, y, "Sensor Data Used")
        y -= 20

        pdf.setFont("Helvetica", 11)
        pdf.drawString(60, y, f"Nitrogen: {sensor_data['nitrogen']} mg/kg")
        y -= 18
        pdf.drawString(60, y, f"Phosphorus: {sensor_data['phosphorus']} mg/kg")
        y -= 18
        pdf.drawString(60, y, f"Potassium: {sensor_data['potassium']} mg/kg")
        y -= 18
        pdf.drawString(60, y, f"Live Connected: {'Yes' if sensor_data['live_connected'] else 'No'}")
        y -= 18
        pdf.drawString(60, y, f"Last Sensor Update: {sensor_data['live_last_update'] or 'N/A'}")
        y -= 30

    # Daily predictions table
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Daily Predictions")
    y -= 20

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(50, y, "Date")
    pdf.drawString(180, y, "Price (LKR/kg)")
    pdf.drawString(320, y, "Demand (Tons)")
    y -= 15

    pdf.setFont("Helvetica", 10)

    for i in range(len(results['dates'])):
        if y < 60:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(50, y, "Date")
            pdf.drawString(180, y, "Price (LKR/kg)")
            pdf.drawString(320, y, "Demand (Tons)")
            y -= 15
            pdf.setFont("Helvetica", 10)

        pdf.drawString(50, y, str(results['dates'][i]))
        pdf.drawString(180, y, f"{results['prices'][i]:.2f}")
        pdf.drawString(320, y, f"{results['demands'][i]:.0f}")
        y -= 15

    pdf.save()
    buffer.seek(0)

    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=paddy_prediction_report.pdf'
    return response
@app.route('/api/latest-sensor', methods=['GET'])
def api_latest_sensor():
    return jsonify({
        "success": True,
        "sensor": sensor_monitor.get_latest()
    })


@app.route('/api/reconnect-sensor', methods=['POST'])
def api_reconnect_sensor():
    try:
        sensor_monitor.stop()
        time.sleep(1)
        sensor_monitor.start()
        return jsonify({
            "success": True,
            "sensor": sensor_monitor.get_latest()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/debug-sensor')
def debug_sensor():
    return jsonify(sensor_monitor.get_latest())


@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


if __name__ == '__main__':
    if should_initialize():
        initialize_app()

    print("\nStarting Flask application...")
    print("Access the application at: http://localhost:5000")
    print("Press CTRL+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)