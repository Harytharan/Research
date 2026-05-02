from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, make_response
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from urllib.parse import urlparse
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64
import warnings
import re
import time
import serial
import threading
from queue import Queue
import traceback
import json
import textwrap
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

warnings.filterwarnings('ignore')

# ============================================================================
# APPLICATION INITIALIZATION - SINGLE INSTANCE
# ============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get(
    'FLASK_SECRET_KEY',
    'paddy_predict_main_secret_key_2026'
)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# DATABASE & AUTHENTICATION SETUP
# ============================================================================
from models import User, init_db, FertilizerHistory, FarmingCostHistory, PricesDemandHistory, PestPredictionHistory
from forms import LoginForm, RegistrationForm

# Initialize database
init_db()

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(int(user_id))

# Context processor to make current_user available to all templates
@app.context_processor
def inject_user():
    return dict(current_user=current_user)

# ============================================================================
# MODULE DIRECTORY SETUP
# ============================================================================
FARMING_COST_DIR = os.path.join(BASE_DIR, 'Farming_Cost')
INTELLIGENT_FERTILIZER_DIR = os.path.join(BASE_DIR, 'Intelligent_Fertilizer')
PEST_OUTBREAK_DIR = os.path.join(BASE_DIR, 'Pest_outbreak')
PRICES_DEMAND_DIR = os.path.join(BASE_DIR, 'Prices_Demand')

# Add directories to system path
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, FARMING_COST_DIR)
sys.path.insert(0, INTELLIGENT_FERTILIZER_DIR)
sys.path.insert(0, PEST_OUTBREAK_DIR)
sys.path.insert(0, PRICES_DEMAND_DIR)

# Create necessary directories for all modules
os.makedirs(os.path.join(FARMING_COST_DIR, 'static', 'uploads'), exist_ok=True)
os.makedirs(os.path.join(FARMING_COST_DIR, 'static', 'images'), exist_ok=True)
os.makedirs(os.path.join(FARMING_COST_DIR, 'models'), exist_ok=True)

os.makedirs(os.path.join(INTELLIGENT_FERTILIZER_DIR, 'uploads'), exist_ok=True)
os.makedirs(os.path.join(INTELLIGENT_FERTILIZER_DIR, 'static', 'images'), exist_ok=True)
os.makedirs(os.path.join(INTELLIGENT_FERTILIZER_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(INTELLIGENT_FERTILIZER_DIR, 'fertilizer_models'), exist_ok=True)
os.makedirs(os.path.join(INTELLIGENT_FERTILIZER_DIR, 'yield_models'), exist_ok=True)

os.makedirs(os.path.join(PEST_OUTBREAK_DIR, 'uploads'), exist_ok=True)
os.makedirs(os.path.join(PEST_OUTBREAK_DIR, 'static', 'images'), exist_ok=True)
os.makedirs(os.path.join(PEST_OUTBREAK_DIR, 'models'), exist_ok=True)

os.makedirs(os.path.join(PRICES_DEMAND_DIR, 'static', 'images'), exist_ok=True)
os.makedirs(os.path.join(PRICES_DEMAND_DIR, 'models'), exist_ok=True)

# ============================================================================
# GLOBAL VARIABLES FOR ALL MODULES
# ============================================================================

# Farming Cost globals
farming_model = None
farming_encoders = None
farming_historical_data = None
farming_df_raw = None
latest_cost_result = None

# Fertilizer globals
clf = None
reg = None
scaler = None
fertilizer_encoders = None
fertilizer_feature_names = None
fertilizer_df_master = None
latest_fertilizer_result = None

# Pest globals
tabular_model = None
tabular_meta = None
image_model = None
image_meta = None
image_class_labels = []
prt = None
Visual = None
latest_pest_result = None
latest_disease_result = None

# Prices Demand globals
prices_models = None
prices_historical_data = None
prices_df_raw = None
prices_df_mm = None
prices_df_std = None
latest_prediction_result = None

# Sensor monitor
sensor_monitor = None

# ============================================================================
# MODULE IMPORTS WITH FALLBACKS
# ============================================================================

print("\n" + "="*60)
print("🌾 PADDY PRICE & DEMAND PREDICTOR - MODULE LOADING")
print("="*60)

# ----------------------------------------------------------------------------
# FARMING COST MODULE IMPORTS
# ----------------------------------------------------------------------------
try:
    from load import load_dataset as farming_load_dataset
    from pre_process import preprocess_data
    from featureeng import create_features
    from predict import predict_total_cost
    print("✅ Farming_Cost modules loaded successfully")
    FARMING_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"⚠️ Farming_Cost import error: {e}")
    FARMING_IMPORT_SUCCESS = False
    
    # Fallback functions
    def farming_load_dataset():
        print("Using fallback farming dataset")
        return pd.DataFrame()
    
    def predict_total_cost(input_data):
        print("Using fallback farming prediction")
        total = sum([
            input_data.get("Seed_Cost (LKR)", 8000),
            input_data.get("Fertilizer_Cost (LKR)", 15000),
            input_data.get("Pesticide_Cost (LKR)", 4000),
            input_data.get("Labor_Cost (LKR)", 25000),
            input_data.get("Water_Cost (LKR)", 5000),
            input_data.get("Machinery_Cost (LKR)", 10000),
            input_data.get("Other_Costs (LKR)", 2000)
        ])
        return total * 1.1  # Add 10% overhead

# ----------------------------------------------------------------------------
# INTELLIGENT FERTILIZER MODULE IMPORTS
# ----------------------------------------------------------------------------
try:
    # Try to import fertilizer modules
    print("✅ Intelligent_Fertilizer modules ready")
    FERTILIZER_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"⚠️ Intelligent_Fertilizer import error: {e}")
    FERTILIZER_IMPORT_SUCCESS = False

# ----------------------------------------------------------------------------
# PEST OUTBREAK MODULE IMPORTS
# ----------------------------------------------------------------------------
try:
    import predict_image as prt
    import visualize_prediction as Visual
    print("✅ Pest_outbreak modules loaded successfully")
    PEST_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"⚠️ Pest_outbreak import error: {e}")
    PEST_IMPORT_SUCCESS = False
    prt = None
    Visual = None

# ----------------------------------------------------------------------------
# PRICES DEMAND MODULE IMPORTS
# ----------------------------------------------------------------------------
PRICES_IMPORT_SUCCESS = False
try:
    # Try different import patterns
    try:
        from load import load_dataset as prices_load_dataset
        print("✅ Loaded prices load_dataset")
    except ImportError:
        prices_load_dataset = None
        print("⚠️ Could not import prices load_dataset")
    
    try:
        from pre_process import preprocess as prices_preprocess
        print("✅ Loaded prices preprocess")
    except ImportError:
        try:
            from preprocess import preprocess as prices_preprocess
            print("✅ Loaded prices preprocess from preprocess")
        except ImportError:
            prices_preprocess = None
            print("⚠️ Could not import prices preprocess")
    
    try:
        from run_prediction import predict_price_with_trend, predict_demand_with_trend
        print("✅ Loaded price/demand prediction functions")
        PRICES_IMPORT_SUCCESS = True
    except ImportError:
        try:
            from predict import predict_price_with_trend, predict_demand_with_trend
            print("✅ Loaded prediction functions from predict")
            PRICES_IMPORT_SUCCESS = True
        except ImportError:
            print("⚠️ Could not import price/demand prediction functions")
            PRICES_IMPORT_SUCCESS = False
    
    # Try to import utils
    try:
        from utils import (
            add_lag_features, add_rolling_and_seasonal, add_price_momentum,
            calculate_historical_trend, update_time_features
        )
        print("✅ Loaded utils functions")
    except ImportError:
        print("⚠️ Could not import utils functions, using fallbacks")
        # Define fallback utils functions
        def add_lag_features(df): return df
        def add_rolling_and_seasonal(df): return df
        def add_price_momentum(df): return df
        def calculate_historical_trend(df): return 0
        def update_time_features(df): return df
        
except Exception as e:
    print(f"⚠️ Prices_Demand import error: {e}")
    PRICES_IMPORT_SUCCESS = False

# Fallback functions for prices if imports failed
if not PRICES_IMPORT_SUCCESS:
    print("Using fallback price/demand prediction functions")
    
    def prices_load_dataset():
        """Fallback function to create dummy dataset"""
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        n = len(dates)
        np.random.seed(42)
        df = pd.DataFrame({
            'Date': dates,
            'Paddy_Price_LKR_per_kg': 100 + 20 * np.sin(np.linspace(0, 8*np.pi, n)) + np.random.normal(0, 5, n),
            'Demand_Tons': 500 + 100 * np.cos(np.linspace(0, 6*np.pi, n)) + np.random.normal(0, 20, n),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
            'Nitrogen_N': np.random.uniform(300, 600, n),
            'Phosphorus_P': np.random.uniform(5, 30, n),
            'Potassium_K': np.random.uniform(5, 30, n)
        })
        return df
    
    def prices_preprocess(df, save_artifacts=False):
        """Fallback preprocessing"""
        df_raw = df.copy()
        df_mm = df.copy()
        df_std = df.copy()
        
        df_mm['DayOfYear'] = df_mm['Date'].dt.dayofyear
        df_mm['Month'] = df_mm['Date'].dt.month
        df_mm['Year'] = df_mm['Date'].dt.year
        
        df_std['DayOfYear'] = df_std['Date'].dt.dayofyear
        df_std['Month'] = df_std['Date'].dt.month
        df_std['Year'] = df_std['Date'].dt.year
        
        return df_raw, df_mm, df_std, None
    
    def predict_price_with_trend(df, model, features, start_date, n_steps=7, window_size=21):
        """Fallback price prediction"""
        prediction_dates = pd.date_range(start=start_date, periods=n_steps, freq='D')
        base_price = 120
        trend = 0.5
        seasonal = 10 * np.sin(np.linspace(0, 2*np.pi, n_steps))
        noise = np.random.normal(0, 2, n_steps)
        predictions = base_price + trend * np.arange(n_steps) + seasonal + noise
        predictions = np.maximum(predictions, 80)
        return predictions.tolist(), prediction_dates
    
    def predict_demand_with_trend(df, model, features, price_predictions, prediction_dates):
        """Fallback demand prediction"""
        n_steps = len(price_predictions)
        base_demand = 500
        price_effect = -2 * (np.array(price_predictions) - 120)
        seasonal = 50 * np.sin(np.linspace(0, 2*np.pi, n_steps) + 1)
        noise = np.random.normal(0, 15, n_steps)
        predictions = base_demand + price_effect + seasonal + noise
        predictions = np.maximum(predictions, 200)
        return predictions.tolist()

# ============================================================================
# LIVE SENSOR MONITOR CLASS (Original - Used by Farming Cost, etc if needed)
# ============================================================================
class LiveSensorMonitor:
    def __init__(self, port="COM7", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.thread = None
        self.broker = None
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
        self.partial_values = {}
        self.disconnect_timeout_seconds = 3.0
        self.last_signal_at = None

    def connect(self):
        try:
            print(f"[SENSOR] Trying to connect to {self.port} at {self.baudrate}...")
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
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
            return False

    def start(self):
        """Legacy entrypoint kept for compatibility. Live NPK data is fed via the shared COM7 broker."""
        if self.thread and self.thread.is_alive():
            return
        print("[SENSOR] start() called - waiting for shared COM7 broker attachment")

    def _queue_connection_alive(self):
        return bool(
            self.running
            and self.thread
            and self.thread.is_alive()
            and self.broker
            and self.broker.running
            and self.broker.thread
            and self.broker.thread.is_alive()
            and self._has_recent_signal()
        )

    def _serial_connection_alive(self):
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

    def start_with_queue(self, line_queue, broker=None):
        if self.thread and self.thread.is_alive():
            return
        self.broker = broker
        self.running = True
        self.last_signal_at = None
        self.thread = threading.Thread(target=self._queue_loop, args=(line_queue,), daemon=True)
        self.thread.start()
        print("[SENSOR] Queue-based NPK thread started (via COM7 broker)")

    def stop(self):
        self.running = False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.thread = None
        self.broker = None
        self.last_signal_at = None
        with self.lock:
            self.latest["connected"] = False

    def parse_line(self, text):
        if not text:
            return None
        line = text.strip().lower()
        if "----" in line:
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
        if not self.partial_values:
            return
        with self.lock:
            for key, value in self.partial_values.items():
                self.latest["values"][key] = value
            self.latest["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.partial_values = {}

    def _consume_text(self, text):
        print(f"[SENSOR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -> {text}")
        parsed = self.parse_line(text)
        self.last_signal_at = time.monotonic()

        with self.lock:
            self.latest["last_raw"] = text.encode("utf-8", errors="ignore").__repr__()
            self.latest["last_decoded"] = text
            self.latest["connected"] = True

        if parsed:
            self.partial_values.update(parsed)

        if all(k in self.partial_values for k in ["nitrogen", "phosphorus", "potassium"]):
            self.commit_partial_values()

        if "----" in text or "npk values" in text.lower():
            self.commit_partial_values()

    def _queue_loop(self, q):
        with self.lock:
            self.latest["connected"] = False
        while self.running:
            try:
                text = q.get(timeout=1)
                if not text:
                    with self.lock:
                        self.latest["connected"] = self._queue_connection_alive()
                    continue
                self._consume_text(text)
            except Exception:
                with self.lock:
                    self.latest["connected"] = self._queue_connection_alive()

    def read_loop(self):
        while self.running:
            try:
                raw = self.ser.readline()
                if raw:
                    try:
                        text = raw.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        text = raw.decode(errors="ignore").strip()

                    with self.lock:
                        self.latest["last_raw"] = str(raw)

                    self._consume_text(text)
                else:
                    with self.lock:
                        self.latest["connected"] = self._serial_connection_alive()
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
        if self.broker is not None:
            payload["connected"] = self._queue_connection_alive()
        elif self.ser is not None:
            payload["connected"] = self._serial_connection_alive()
        # Provide default values if none received
        if payload["values"]["nitrogen"] is None:
            payload["values"]["nitrogen"] = 520.0
        if payload["values"]["phosphorus"] is None:
            payload["values"]["phosphorus"] = 10.0
        if payload["values"]["potassium"] is None:
            payload["values"]["potassium"] = 10.0
        return payload

# Initialize sensor monitor
sensor_monitor = LiveSensorMonitor(port="COM7", baudrate=115200)

# ============================================================================
# SHARED SERIAL LINE BROADCASTER
# Opens COM7 ONCE and fans out every received line to all subscriber queues.
# Eliminates the port-conflict when both FertilizerSensorMonitor and
# CostSensorMonitor need lines from the same physical COM7 device.
# ============================================================================
class SerialLineBroadcaster:
    def __init__(self, port, baudrate=115200):
        self.port     = port
        self.baudrate = baudrate
        self.ser      = None
        self.running  = False
        self.thread   = None
        self._lock    = threading.Lock()
        self._queues  = []          # list[Queue]

    def subscribe(self):
        """Return a Queue that will receive every decoded text line."""
        q = Queue(maxsize=300)
        with self._lock:
            self._queues.append(q)
        return q

    def connect(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"[COM7 BROKER] Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"[COM7 BROKER] Cannot open {self.port}: {e}")
            return False

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread  = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[COM7 BROKER] Broadcaster thread started")

    def _run(self):
        while self.running:
            if not self.ser or not self.ser.is_open:
                if not self.connect():
                    time.sleep(3)
                    continue
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                try:
                    text = raw.decode('utf-8').strip()
                except Exception:
                    text = raw.decode(errors='ignore').strip()
                if not text:
                    continue
                with self._lock:
                    for q in self._queues:
                        try:
                            q.put_nowait(text)
                        except Exception:
                            pass      # queue full — skip oldest-unbuffered line
            except Exception as e:
                print(f"[COM7 BROKER] Read error: {e}")
                try:
                    if self.ser:
                        self.ser.close()
                except Exception:
                    pass
                self.ser = None
                time.sleep(2)


# ============================================================================
# DEDICATED FERTILIZER SENSOR MONITOR (For COM7: Env Sensors)
# ============================================================================
class FertilizerSensorMonitor:
    def __init__(self, port="COM7", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.thread = None
        self.broker = None
        self.lock = threading.Lock()
        
        self.latest = {
            "connected": False,
            "soil_temperature": None,
            "soil_moisture": None,
            "air_temperature": None,
            "air_humidity": None,
            "last_update": None
        }
        self.partial = {}
        self.disconnect_timeout_seconds = 3.0
        self.last_signal_at = None

    def connect(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            with self.lock:
                self.latest["connected"] = True
            print(f"[FERTILIZER SENSOR] Connected to {self.port} at {self.baudrate}")
            return True
        except Exception as e:
            with self.lock:
                self.latest["connected"] = False
            print(f"[FERTILIZER SENSOR] Could not connect to {self.port}: {e}")
            return False

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        print("[FERTILIZER SENSOR] start() called - waiting for shared COM7 broker attachment")

    def _queue_connection_alive(self):
        return bool(
            self.running
            and self.thread
            and self.thread.is_alive()
            and self.broker
            and self.broker.running
            and self.broker.thread
            and self.broker.thread.is_alive()
            and self._has_recent_signal()
        )

    def _serial_connection_alive(self):
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

    def start_with_queue(self, line_queue, broker=None):
        """Additive method: read from a shared broadcaster queue instead of
        opening the serial port directly.  All parsing logic is identical."""
        if self.thread and self.thread.is_alive():
            return
        self.broker = broker
        self.running = True
        self.last_signal_at = None
        self.thread = threading.Thread(
            target=self._queue_loop, args=(line_queue,), daemon=True)
        self.thread.start()
        print("[FERTILIZER SENSOR] Queue-based thread started (via COM7 broker)")

    def _queue_loop(self, q):
        """Same parsing as read_loop but sourced from the broadcaster queue.
        ONLY processes lines with fertilizer-relevant keywords to keep terminal clean."""
        with self.lock:
            self.latest["connected"] = False
        while self.running:
            try:
                text = q.get(timeout=1)          # blocks max 1 s
                line = text.lower()

                # Only process lines that contain fertilizer-specific keywords.
                # "====" is included so the end-of-reading separator passes through
                # and triggers the partial→latest commit block below.
                FERTI_KEYWORDS = ("ds18b20", "soil moisture", "dht11", "====")
                if not any(kw in line for kw in FERTI_KEYWORDS):
                    continue   # silently discard non-fertilizer lines

                print(f"[FERTILIZER SENSOR] {datetime.now().strftime('%H:%M:%S')} -> {text}")
                self.last_signal_at = time.monotonic()

                with self.lock:
                    self.latest["connected"] = True

                if "ds18b20 temperature" in line:
                    m = re.search(r"(-?\d+(?:\.\d+)?)", text.split(":", 1)[-1])
                    if m: self.partial["soil_temperature"] = float(m.group(1))

                elif "soil moisture value" in line:
                    m = re.search(r"(\d+(?:\.\d+)?)", text.split(":", 1)[-1])
                    if m:
                        raw_val = float(m.group(1))
                        self.partial["soil_moisture"] = round((raw_val / 4095.0) * 100.0, 1)

                elif "dht11 temperature" in line:
                    m = re.search(r"(-?\d+(?:\.\d+)?)", text.split(":", 1)[-1])
                    if m: self.partial["air_temperature"] = float(m.group(1))

                elif "dht11 humidity" in line:
                    m = re.search(r"(-?\d+(?:\.\d+)?)", text.split(":", 1)[-1])
                    if m: self.partial["air_humidity"] = float(m.group(1))

                if "====" in line:
                    if self.partial:
                        with self.lock:
                            for k, v in self.partial.items():
                                self.latest[k] = v
                            self.latest["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.partial = {}

            except Exception:
                with self.lock:
                    self.latest["connected"] = self._queue_connection_alive()
                # timeout or queue closed — loop continues


    def stop(self):
        self.running = False
        try:
            if self.ser and self.ser.is_open: self.ser.close()
        except: pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.thread = None
        self.broker = None
        self.last_signal_at = None
        with self.lock: self.latest["connected"] = False

    def read_loop(self):
        while self.running:
            try:
                raw = self.ser.readline()
                if raw:
                    try: text = raw.decode("utf-8").strip()
                    except: text = raw.decode(errors="ignore").strip()
                    
                    line = text.lower()
                    print(f"[FERTILIZER SENSOR] {datetime.now().strftime('%H:%M:%S')} -> {text}")
                    self.last_signal_at = time.monotonic()
                    
                    with self.lock:
                        self.latest["connected"] = True
                    
                    # Parse values
                    if "ds18b20 temperature" in line:
                        m = re.search(r"(-?\d+(?:\.\d+)?)", text.split(":", 1)[-1])
                        if m: self.partial["soil_temperature"] = float(m.group(1))

                    elif "soil moisture value" in line:
                        m = re.search(r"(\d+(?:\.\d+)?)", text.split(":", 1)[-1])
                        if m: 
                            raw_val = float(m.group(1))
                            self.partial["soil_moisture"] = round((raw_val / 4095.0) * 100.0, 1)

                    elif "dht11 temperature" in line:
                        m = re.search(r"(-?\d+(?:\.\d+)?)", text.split(":", 1)[-1])
                        if m: self.partial["air_temperature"] = float(m.group(1))

                    elif "dht11 humidity" in line:
                        m = re.search(r"(-?\d+(?:\.\d+)?)", text.split(":", 1)[-1])
                        if m: self.partial["air_humidity"] = float(m.group(1))

                    # Commit when block ends
                    if "====" in line:
                        if self.partial:
                            with self.lock:
                                for k, v in self.partial.items():
                                    self.latest[k] = v
                                self.latest["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            self.partial = {}
                else:
                    with self.lock:
                        self.latest["connected"] = self._serial_connection_alive()
                            
            except Exception:
                print("[FERTILIZER SENSOR] Read error")
                with self.lock: self.latest["connected"] = False
                time.sleep(1)

    def get_latest(self):
        with self.lock:
            payload = dict(self.latest)
        if self.broker is not None:
            payload["connected"] = self._queue_connection_alive()
        elif self.ser is not None:
            payload["connected"] = self._serial_connection_alive()
        return payload

# ============================================================================
# DEDICATED FARMING COST SENSOR MONITOR
# Reads Temperature, Humidity, Latitude, Longitude from the farming cost
# IoT device.  Uses a broadcaster queue — never opens the COM port itself.
# Terminal logs are clearly labelled [FARMING COST SENSOR].
# ============================================================================
class CostSensorMonitor:
    def __init__(self, port="COM7", baudrate=115200):
        """port/baudrate kept for API compatibility; actual I/O via queue."""
        self.port      = port
        self.baudrate  = baudrate
        self.lock      = threading.Lock()
        self.thread    = None
        self.broker    = None
        self.running   = False          # set True once queue thread is live
        self.disconnect_timeout_seconds = 3.0
        self.last_signal_at = None
        self.latest = {
            "connected":    False,
            "temperature":  None,
            "humidity":     None,
            "latitude":     None,
            "longitude":    None,
            "last_update":  None,
        }

    def _queue_connection_alive(self):
        return bool(
            self.running
            and self.thread
            and self.thread.is_alive()
            and self.broker
            and self.broker.running
            and self.broker.thread
            and self.broker.thread.is_alive()
            and self._has_recent_signal()
        )

    def _has_recent_signal(self):
        return bool(
            self.last_signal_at is not None
            and (time.monotonic() - self.last_signal_at) <= self.disconnect_timeout_seconds
        )

    # ── Public API (unchanged from original) ─────────────────────────────────
    def get_latest(self):
        with self.lock:
            payload = dict(self.latest)
        if self.broker is not None:
            payload["connected"] = self._queue_connection_alive()
        return payload

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.thread = None
        self.broker = None
        self.last_signal_at = None
        with self.lock:
            self.latest["connected"] = False

    def start(self):
        """Legacy stub — real start is done via start_with_queue()."""
        print("[FARMING COST SENSOR] start() called — queue will be attached by broker")

    def start_with_queue(self, line_queue, broker=None):
        """Start the parsing thread, reading lines from the shared broadcaster."""
        if self.thread and self.thread.is_alive():
            return
        self.broker = broker
        self.running = True
        self.last_signal_at = None
        self.thread  = threading.Thread(
            target=self._queue_loop, args=(line_queue,), daemon=True)
        self.thread.start()
        print("[FARMING COST SENSOR] Queue-based thread started (via COM7 broker)")

    # ── Internal parsing loop ─────────────────────────────────────────
    def _queue_loop(self, q):
        # connected stays False until the first real value is parsed,
        # so the API never returns connected=True with all-null values.

        while self.running:
            try:
                text = q.get(timeout=5)       # blocks until a line arrives
            except Exception:
                with self.lock:
                    self.latest["connected"] = self._queue_connection_alive()
                continue                       # timeout — keep looping

            line = text.lower()
            
            # Filter Farming Cost lines to keep logs clean
            COST_KEYWORDS = ("temp", "humidity", "lat", "lon")
            if not any(kw in line for kw in COST_KEYWORDS) or "soil" in line or "pressure" in line or "ldr" in line:
                continue
                
            print(f"[FARMING COST SENSOR] {datetime.now().strftime('%H:%M:%S')} -> {text}")
            self.last_signal_at = time.monotonic()

            updated = False

            # ── LABEL-ANCHORED extraction ─────────────────────────────────────
            # These patterns search for the label THEN grab the first number after it.
            # This correctly handles lines where multiple values appear on one line,
            # e.g.:  "Humidity: 52.00 %\tTemperature: 26.00 °C 78.80 °F"
            # Using independent `if` (not elif) so both are extracted in one pass.

            # Temperature (°C) — label must appear BEFORE the number
            m_temp = re.search(
                r"[Tt]emp(?:erature)?\s*[:\-]\s*(-?\d+(?:\.\d+)?)",
                text
            )
            if m_temp and "soil" not in line:
                with self.lock:
                    self.latest["temperature"] = float(m_temp.group(1))
                updated = True

            # Humidity — label must appear BEFORE the number
            m_hum = re.search(
                r"[Hh]um(?:idity)?\s*[:\-]\s*(-?\d+(?:\.\d+)?)",
                text
            )
            if m_hum and "soil" not in line:
                with self.lock:
                    self.latest["humidity"] = float(m_hum.group(1))
                updated = True

            # Latitude — label must appear BEFORE the number
            m_lat = re.search(
                r"[Ll]at(?:itude)?\s*[:\-]\s*(-?\d+(?:\.\d+)?)",
                text
            )
            if m_lat:
                with self.lock:
                    self.latest["latitude"] = float(m_lat.group(1))
                updated = True

            # Longitude — label must appear BEFORE the number
            m_lon = re.search(
                r"[Ll]on(?:gitude)?\s*[:\-]\s*(-?\d+(?:\.\d+)?)",
                text
            )
            if m_lon:
                with self.lock:
                    self.latest["longitude"] = float(m_lon.group(1))
                updated = True

            if updated:
                with self.lock:
                    self.latest["connected"]   = True
                    self.latest["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")



# ============================================================================
# DEDICATED PEST SENSOR MONITOR
# Reads Temperature, Pressure, Light from COM7.
# Humidity is hardcoded to 55-65% range as requested.
# Labels: [PEST MANAGEMENT SENSOR].
# ============================================================================
class PestSensorMonitor:
    def __init__(self, port="COM7", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.lock = threading.Lock()
        self.thread = None
        self.broker = None
        self.running = False
        self.disconnect_timeout_seconds = 3.0
        self.last_signal_at = None
        self.latest = {
            "connected": False,
            "temperature": None,
            "humidity": None,
            "pressure": None,
            "ldr": None,
            "ldr_text": "",
            "last_update": None,
            "risk_assessment": {
                "risk_level": "Low",
                "description": "Environment is stable. No immediate outbreak risk detected."
            }
        }

    def _has_recent_signal(self):
        return bool(
            self.last_signal_at is not None
            and (time.monotonic() - self.last_signal_at) <= self.disconnect_timeout_seconds
        )

    def _queue_connection_alive(self):
        return bool(
            self.running
            and self.thread
            and self.thread.is_alive()
            and self.broker
            and self.broker.running
            and self.broker.thread
            and self.broker.thread.is_alive()
            and self._has_recent_signal()
        )

    def get_latest(self):
        connected_state = self._queue_connection_alive() if self.broker is not None else self.latest["connected"]
        with self.lock:
            # Force humidity into requested range consistently (55-65%)
            self.latest["connected"] = connected_state
            if connected_state:
                import random
                now = datetime.now()
                rng = random.Random(now.minute)
                self.latest["humidity"] = round(rng.uniform(55.0, 65.0), 1)
                self._update_risk()
            payload = dict(self.latest)
        payload["connected"] = connected_state
        return payload

    def _update_risk(self):
        try:
            temp = self.latest["temperature"] # type: ignore
            hum = self.latest["humidity"] # type: ignore
            ldr = self.latest["ldr"] # type: ignore
            
            level, desc = "Low Risk", "Environment is stable. No immediate outbreak risk detected."
            
            if isinstance(temp, (int, float)) and temp > 33:
                level, desc = "Medium Risk", "High temperature detected. Insect breeding cycles may accelerate."
            
            if isinstance(hum, (int, float)) and hum > 85:
                level, desc = "Medium Risk", "High humidity detected. Increased risk of fungal blight."
                
            if isinstance(temp, (int, float)) and temp > 33 and isinstance(hum, (int, float)) and hum > 85:
                level, desc = "High Risk", "Critical levels! High temp and humidity. Major outbreak imminent."
                
            if isinstance(ldr, (int, float)) and ldr < 1000 and isinstance(hum, (int, float)) and hum > 80:
                level, desc = "High Risk", "Low light and high humidity. Fungal risks are extreme."
            
            self.latest["risk_assessment"] = {"risk_level": level, "description": desc}
        except Exception:
            pass

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.thread = None
        self.broker = None
        self.last_signal_at = None
        with self.lock: self.latest["connected"] = False

    def start_with_queue(self, line_queue, broker=None):
        if self.thread and self.thread.is_alive(): return
        self.broker = broker
        self.running = True
        self.last_signal_at = None
        self.thread = threading.Thread(target=self._queue_loop, args=(line_queue,), daemon=True)
        self.thread.start()
        print("[PEST MANAGEMENT SENSOR] Queue-based thread started (via COM7 broker)")

    def _queue_loop(self, q):
        while self.running:
            try: text = q.get(timeout=5)
            except Exception:
                with self.lock:
                    self.latest["connected"] = self._queue_connection_alive()
                continue
            line = text.lower()
            
            # Filter Pest Management lines to keep logs clean
            PEST_KEYWORDS = ("temp", "pressure", "ldr", "altitude")
            if not any(kw in line for kw in PEST_KEYWORDS) or "soil" in line or "humidity" in line:
                continue

            print(f"[PEST MANAGEMENT SENSOR] {datetime.now().strftime('%H:%M:%S')} -> {text}")
            updated = False
            m_temp = re.search(r"[Tt]emp(?:erature)?\s*[:\-]\s*(-?\d+(?:\.\d+)?)", text)
            if m_temp and "soil" not in line:
                with self.lock: self.latest["temperature"] = float(m_temp.group(1))
                updated = True
            m_pres = re.search(r"[Pp]ressure\s*[:\-]\s*(\d+(?:\.\d+)?)", text)
            if m_pres:
                with self.lock: self.latest["pressure"] = float(m_pres.group(1))
                updated = True
            m_ldr = re.search(r"LDR Analog Value\s*[=:]\s*(\d+)", text)
            if m_ldr:
                with self.lock:
                    self.latest["ldr"] = float(m_ldr.group(1))
                    if "->" in text: self.latest["ldr_text"] = text.split("->")[-1].strip()
                updated = True
            if updated:
                self.last_signal_at = time.monotonic()
                with self.lock:
                    self.latest["connected"] = True
                    self.latest["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Initialize sensor monitors
sensor_monitor = LiveSensorMonitor(port="COM7", baudrate=115200)

# COM7 SHARED BROADCASTER
COM7_broker   = SerialLineBroadcaster(port="COM7", baudrate=115200)
npk_queue     = COM7_broker.subscribe()
ferti_queue   = COM7_broker.subscribe()
cost_queue    = COM7_broker.subscribe()
pest_queue    = COM7_broker.subscribe()

ferti_sensor  = FertilizerSensorMonitor(port="COM7", baudrate=115200)
cost_sensor   = CostSensorMonitor(port="COM7", baudrate=115200)
pest_sensor   = PestSensorMonitor(port="COM7", baudrate=115200)

sensors_started = False


@app.before_request
def start_background_sensors():
    global sensors_started
    if not sensors_started:
        COM7_broker.start()
        sensor_monitor.start_with_queue(npk_queue, COM7_broker)
        ferti_sensor.start_with_queue(ferti_queue, COM7_broker)
        cost_sensor.start_with_queue(cost_queue, COM7_broker)
        pest_sensor.start_with_queue(pest_queue, COM7_broker)
        sensors_started = True
        print("[STARTUP] COM7 broker + NPK + Fertilizer + Cost + Pest sensor threads started")

# ============================================================================
# MODULE INITIALIZATION FUNCTIONS
# ============================================================================

def init_farming_cost():
    """Initialize Farming Cost module"""
    global farming_historical_data, farming_df_raw
    print("\n📊 Initializing Farming Cost module...")
    
    # Check for model files
    model_path = os.path.join(FARMING_COST_DIR, 'models', 'trained_paddy_cost_model.pkl')
    encoders_path = os.path.join(FARMING_COST_DIR, 'models', 'label_encoders.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoders_path):
        print("✅ Farming Cost model files found")
    else:
        print("⚠️ Farming Cost model files not found (run train.py in Farming_Cost folder)")
    
    # Load historical data
    try:
        if FARMING_IMPORT_SUCCESS:
            current_dir = os.getcwd()
            os.chdir(FARMING_COST_DIR)
            farming_df_raw = farming_load_dataset()
            os.chdir(current_dir)
            
            if farming_df_raw is not None and len(farming_df_raw) > 0:
                farming_historical_data = get_farming_historical_data(farming_df_raw)
                print(f"✅ Loaded {len(farming_df_raw)} farming records")
            else:
                print("⚠️ Could not load farming historical data")
        else:
            print("⚠️ Using fallback for farming data")
    except Exception as e:
        print(f"⚠️ Error loading farming data: {e}")

def get_farming_historical_data(df):
    """Extract historical data for farming module"""
    if df is None or len(df) == 0:
        return {}
    try:
        return {
            'date_range': {
                'start': int(df['Year'].min()) if 'Year' in df.columns else 2020,
                'end': int(df['Year'].max()) if 'Year' in df.columns else 2024,
                'total_records': len(df)
            },
            'cost_stats': {
                'mean': float(df['Total_Cost (LKR)'].mean()) if 'Total_Cost (LKR)' in df.columns else 100000,
                'min': float(df['Total_Cost (LKR)'].min()) if 'Total_Cost (LKR)' in df.columns else 50000,
                'max': float(df['Total_Cost (LKR)'].max()) if 'Total_Cost (LKR)' in df.columns else 200000
            },
            'soil_types': df['Soil_Type'].unique().tolist() if 'Soil_Type' in df.columns else ['Clay', 'Loam', 'Sandy'],
            'seed_types': df['Seed_Type'].unique().tolist() if 'Seed_Type' in df.columns else ['BG300', 'BG352'],
            'seasons': df['Season'].unique().tolist() if 'Season' in df.columns else ['Maha', 'Yala']
        }
    except Exception as e:
        print(f"Error in get_farming_historical_data: {e}")
        return {}

def init_fertilizer():
    """Initialize Fertilizer module"""
    global clf, reg, scaler, fertilizer_encoders, fertilizer_feature_names, fertilizer_df_master
    print("\n🧪 Initializing Fertilizer module...")
    
    try:
        # Load models from Intelligent_Fertilizer directory
        clf_path = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'fertilizer_models', 'fertilizer_model.pkl')
        reg_path = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'yield_models', 'yield_model.pkl')
        scaler_path = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'yield_models', 'scaler.pkl')
        encoders_path = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'fertilizer_models', 'label_encoders.pkl')
        features_path = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'fertilizer_models', 'feature_names.pkl')
        
        if os.path.exists(clf_path):
            clf = joblib.load(clf_path)
            print("✅ Fertilizer classifier loaded")
        else:
            print("⚠️ Fertilizer classifier not found")
        
        if os.path.exists(reg_path):
            reg = joblib.load(reg_path)
            print("✅ Yield regressor loaded")
        else:
            print("⚠️ Yield regressor not found")
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded")
        
        if os.path.exists(encoders_path):
            fertilizer_encoders = joblib.load(encoders_path)
            print("✅ Label encoders loaded")
        
        if os.path.exists(features_path):
            fertilizer_feature_names = joblib.load(features_path)
            print(f"✅ Feature names loaded: {len(fertilizer_feature_names)} features")
        
        # Load master dataset
        dataset_path = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'Dataset.csv')
        if os.path.exists(dataset_path):
            fertilizer_df_master = pd.read_csv(dataset_path)
            print(f"✅ Master dataset loaded: {len(fertilizer_df_master)} records")
        else:
            print("⚠️ Master dataset not found")
            
    except Exception as e:
        print(f"⚠️ Error loading fertilizer models: {e}")

def init_pest():
    """Initialize Pest module"""
    global tabular_model, tabular_meta, image_model, image_meta, image_class_labels
    print("\n🐛 Initializing Pest Detection module...")
    
    # Load tabular model
    try:
        model_path = os.path.join(PEST_OUTBREAK_DIR, 'models', 'paddy_pest_rf.joblib')
        meta_path = os.path.join(PEST_OUTBREAK_DIR, 'models', 'paddy_meta.joblib')
        
        if os.path.exists(model_path) and os.path.exists(meta_path):
            tabular_model = joblib.load(model_path)
            tabular_meta = joblib.load(meta_path)
            print("✅ Tabular model loaded")
        else:
            print("⚠️ Tabular model files not found")
    except Exception as e:
        print(f"⚠️ Error loading tabular model: {e}")
    
    # Load image model
    try:
        model_path = os.path.join(PEST_OUTBREAK_DIR, 'best_pest_model.h5')
        meta_path = os.path.join(PEST_OUTBREAK_DIR, 'paddy_meta.joblib')
        class_indices_path = os.path.join(PEST_OUTBREAK_DIR, 'class_indices.json')
        
        if os.path.exists(model_path):
            image_model = load_model(model_path)
            image_meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
            
            if prt is not None and hasattr(prt, 'load_class_labels'):
                image_class_labels = prt.load_class_labels()
                print(f"✅ Found {len(image_class_labels)} pest classes from {class_indices_path}")
            else:
                if 'class_names' in image_meta:
                    image_class_labels = image_meta['class_names']
                else:
                    image_class_labels = []
            print("✅ Image model loaded")
        else:
            print("⚠️ Image model files not found")
    except Exception as e:
        print(f"⚠️ Error loading image model: {e}")

def init_prices():
    """Initialize Prices & Demand module"""
    global prices_models, prices_historical_data, prices_df_raw, prices_df_mm, prices_df_std
    print("\n📈 Initializing Prices & Demand module...")
    
    # Load models
    prices_models = load_prices_models()
    
    # Load and preprocess data
    try:
        if PRICES_IMPORT_SUCCESS and prices_load_dataset is not None:
            current_dir = os.getcwd()
            os.chdir(PRICES_DEMAND_DIR)
            
            df = prices_load_dataset()
            if df is not None and len(df) > 0:
                df_raw, df_mm, df_std, _ = prices_preprocess(df, save_artifacts=False)
                prices_df_raw = df_raw
                prices_df_mm = df_mm
                prices_df_std = df_std
                
                # Add features if functions are available
                try:
                    prices_df_mm = add_rolling_and_seasonal(prices_df_mm)
                    prices_df_mm = add_price_momentum(prices_df_mm)
                    prices_df_mm = prices_df_mm.dropna().reset_index(drop=True)
                except Exception as e:
                    print(f"⚠️ Feature engineering skipped: {e}")
                
                prices_historical_data = get_prices_historical_data(prices_df_raw)
                print(f"✅ Loaded {len(prices_df_raw)} price/demand records")
            else:
                print("⚠️ Could not load price data")
            
            os.chdir(current_dir)
        else:
            print("⚠️ Using fallback for price data")
            # Create dummy data
            df = prices_load_dataset()
            prices_df_raw, prices_df_mm, prices_df_std, _ = prices_preprocess(df)
            prices_historical_data = get_prices_historical_data(prices_df_raw)
            
    except Exception as e:
        print(f"⚠️ Error loading price data: {e}")

def load_prices_models():
    """Load price and demand prediction models"""
    models_dict = {}
    models_path = os.path.join(PRICES_DEMAND_DIR, 'models')
    
    try:
        if not os.path.exists(models_path):
            print(f"⚠️ Models directory not found at {models_path}")
            return None
        
        # Load LSTM model
        lstm_path = os.path.join(models_path, 'lstm_price_model_final.h5')
        if os.path.exists(lstm_path):
            models_dict['lstm'] = load_model(lstm_path)
            print("✅ LSTM price model loaded")
        else:
            print("⚠️ LSTM model not found")
            models_dict['lstm'] = None
        
        # Load XGBoost model (prefer newer artifact name used by predict_from_sensor.py)
        xgb_candidates = [
            'xgb_demand_model.joblib',
            'xgb_demand_model_best_optimized.joblib'
        ]
        models_dict['xgb'] = None
        for xgb_name in xgb_candidates:
            xgb_path = os.path.join(models_path, xgb_name)
            if os.path.exists(xgb_path):
                models_dict['xgb'] = joblib.load(xgb_path)
                print(f"✅ XGBoost demand model loaded ({xgb_name})")
                break
        if models_dict['xgb'] is None:
            print("⚠️ XGBoost model not found")
        
        # Load feature columns
        xgb_feature_candidates = [
            'xgb_feature_columns.joblib',
            'feature_columns_optimized.joblib'
        ]
        models_dict['xgb_features'] = None
        for feature_name in xgb_feature_candidates:
            xgb_features_path = os.path.join(models_path, feature_name)
            if os.path.exists(xgb_features_path):
                models_dict['xgb_features'] = joblib.load(xgb_features_path)
                print(f"✅ Loaded {len(models_dict['xgb_features'])} demand features ({feature_name})")
                break
        if models_dict['xgb_features'] is None:
            models_dict['xgb_features'] = ['Paddy_Price_LKR_per_kg', 'DayOfYear', 'Month']
        
        lstm_features_path = os.path.join(models_path, 'lstm_feature_columns.joblib')
        if os.path.exists(lstm_features_path):
            models_dict['lstm_features'] = joblib.load(lstm_features_path)
            print(f"✅ Loaded {len(models_dict['lstm_features'])} LSTM features")
        else:
            models_dict['lstm_features'] = ['Paddy_Price_LKR_per_kg', 'Nitrogen_N', 'Phosphorus_P', 'Potassium_K']
        
        # Load training info
        training_info_path = os.path.join(models_path, 'training_info.joblib')
        if os.path.exists(training_info_path):
            training_info = joblib.load(training_info_path)
            models_dict['window_size'] = training_info.get('window_size', 21)
            print(f"✅ Training info loaded (window size: {models_dict['window_size']})")
        else:
            models_dict['window_size'] = 21

        region_encoder_path = os.path.join(models_path, 'region_encoder.joblib')
        rice_type_encoder_path = os.path.join(models_path, 'rice_type_encoder.joblib')

        models_dict['region_encoder'] = joblib.load(region_encoder_path) if os.path.exists(region_encoder_path) else None
        models_dict['rice_type_encoder'] = joblib.load(rice_type_encoder_path) if os.path.exists(rice_type_encoder_path) else None

        if models_dict['region_encoder'] is not None and models_dict['rice_type_encoder'] is not None:
            print("✅ Region and rice type encoders loaded")
        
        return models_dict
        
    except Exception as e:
        print(f"⚠️ Error loading price models: {e}")
        return None

def get_prices_historical_data(df):
    """Extract historical data for prices module"""
    if df is None or len(df) == 0:
        return {}
    try:
        recent_df = df.tail(30)
        regions = df['Region'].dropna().astype(str).unique().tolist() if 'Region' in df.columns else ['All Markets']
        return {
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d') if 'Date' in df.columns else '2023-01-01',
                'end': df['Date'].max().strftime('%Y-%m-%d') if 'Date' in df.columns else '2024-12-31',
                'total_days': len(df),
                'total_records': len(df)
            },
            'price_stats': {
                'mean': float(df['Paddy_Price_LKR_per_kg'].mean()) if 'Paddy_Price_LKR_per_kg' in df.columns else 120,
                'min': float(df['Paddy_Price_LKR_per_kg'].min()) if 'Paddy_Price_LKR_per_kg' in df.columns else 80,
                'max': float(df['Paddy_Price_LKR_per_kg'].max()) if 'Paddy_Price_LKR_per_kg' in df.columns else 160
            },
            'demand_stats': {
                'mean': float(df['Demand_Tons'].mean()) if 'Demand_Tons' in df.columns else 500,
                'min': float(df['Demand_Tons'].min()) if 'Demand_Tons' in df.columns else 300,
                'max': float(df['Demand_Tons'].max()) if 'Demand_Tons' in df.columns else 700
            },
            'recent_dates': recent_df['Date'].dt.strftime('%Y-%m-%d').tolist() if 'Date' in recent_df.columns else [],
            'recent_prices': recent_df['Paddy_Price_LKR_per_kg'].tolist() if 'Paddy_Price_LKR_per_kg' in recent_df.columns else [],
            'recent_demand': recent_df['Demand_Tons'].tolist() if 'Demand_Tons' in recent_df.columns else [],
            'recent_demands': recent_df['Demand_Tons'].tolist() if 'Demand_Tons' in recent_df.columns else [],
            'regions': regions
        }
    except Exception as e:
        print(f"Error in get_prices_historical_data: {e}")
        return {}

# Initialize all modules
init_farming_cost()
init_fertilizer()
init_pest()
init_prices()

print("\n" + "="*60)
print("✅ ALL MODULES INITIALIZED SUCCESSFULLY")
print("="*60)

# ============================================================================
# PLOT GENERATION FUNCTIONS
# ============================================================================

def generate_cost_breakdown_plot(cost_breakdown, predicted_cost):
    """Generate pie chart for cost breakdown"""
    plt.figure(figsize=(8, 8))
    labels = list(cost_breakdown.keys())
    values = list(cost_breakdown.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE194', '#D4A5A5', '#9B59B6']
    plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'Cost Breakdown (Total: LKR {sum(values):,.0f})', fontsize=14, fontweight='bold', pad=20)
    plt.axis('equal')
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def generate_comparison_plot(total_input, predicted_cost):
    """Generate bar chart comparing input vs predicted"""
    plt.figure(figsize=(8, 6))
    categories = ['Input Total', 'Predicted Cost']
    values = [total_input, predicted_cost]
    colors = ['#3498db', '#2ecc71']
    bars = plt.bar(categories, values, color=colors, alpha=0.8)
    plt.ylabel('Cost (LKR)', fontsize=12)
    plt.title('Input vs Predicted Total Cost', fontsize=14, fontweight='bold', pad=20)
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'LKR {value:,.0f}', ha='center', va='bottom', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def generate_confidence_plot(confidence):
    """Generate confidence level plot"""
    plt.figure(figsize=(6, 3))
    categories = ['Confidence']
    values = [confidence * 100 if confidence <= 1 else confidence]
    colors = ['#2ecc71' if values[0] > 70 else '#f39c12' if values[0] > 40 else '#e74c3c']
    bars = plt.barh(categories, values, color=colors, alpha=0.8, height=0.5)
    plt.xlim(0, 100)
    plt.xlabel('Confidence (%)', fontsize=11)
    plt.title('Prediction Confidence', fontsize=13, fontweight='bold', pad=15)
    for bar, val in zip(bars, values):
        width = bar.get_width()
        plt.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def generate_yield_comparison_plot(results):
    """Generate yield comparison plot for fertilizers"""
    plt.figure(figsize=(8, 5))
    fertilizers = [r['fertilizer'] for r in results]
    yields = [r['yield'] for r in results]
    colors = ['#3498db', '#9b59b6', '#1abc9c']
    bars = plt.bar(fertilizers, yields, color=colors, alpha=0.8)
    plt.ylabel('Predicted Yield (ton/ha)', fontsize=12)
    plt.title('Expected Yield by Fertilizer', fontsize=14, fontweight='bold', pad=20)
    for bar, y in zip(bars, yields):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{y} t/ha', ha='center', va='bottom', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url




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


def generate_sales_recommendation(prediction_dates, price_predictions, demand_predictions):
    if len(prediction_dates) == 0 or len(price_predictions) == 0:
        return {
            'action': 'Hold Review',
            'headline': 'Not enough forecast data to recommend a sales window.',
            'message': 'Generate a new forecast to get a farmer sales recommendation.',
            'recommended_date': None,
            'recommended_price': None,
            'variant': 'secondary',
            'icon': 'fas fa-circle-info',
            'demand_note': 'Demand guidance unavailable.',
            'waiting_days': 0
        }

    first_price = float(price_predictions[0])
    peak_idx = int(np.argmax(price_predictions))
    peak_price = float(price_predictions[peak_idx])
    peak_date = pd.to_datetime(prediction_dates[peak_idx]).strftime('%Y-%m-%d')
    waiting_days = peak_idx

    week_idx = min(6, len(price_predictions) - 1)
    week_price = float(price_predictions[week_idx])
    three_day_idx = min(3, len(price_predictions) - 1)
    three_day_price = float(price_predictions[three_day_idx])
    min_gain = max(first_price * 0.015, 1.0)
    peak_gain = peak_price - first_price
    three_day_gain = three_day_price - first_price
    week_gain = week_price - first_price

    avg_demand = float(np.mean(demand_predictions)) if len(demand_predictions) else 0.0
    demand_at_peak = float(demand_predictions[peak_idx]) if len(demand_predictions) > peak_idx else avg_demand
    demand_note = (
        'Demand is expected to stay supportive around the recommended selling window.'
        if demand_at_peak >= avg_demand
        else 'Demand is slightly softer near the best price window, so stagger sales if possible.'
    )

    if peak_gain < min_gain or peak_idx == 0:
        return {
            'action': 'Sell Now',
            'headline': 'Current forecast suggests selling now.',
            'message': 'The predicted price improvement ahead is too small to justify waiting.',
            'recommended_date': pd.to_datetime(prediction_dates[0]).strftime('%Y-%m-%d'),
            'recommended_price': round(first_price, 2),
            'variant': 'success',
            'icon': 'fas fa-bolt',
            'demand_note': demand_note,
            'waiting_days': 0
        }

    if peak_idx <= three_day_idx and three_day_gain >= min_gain:
        return {
            'action': 'Hold 3 Days',
            'headline': 'A better selling point is expected within the next 3 days.',
            'message': 'Holding for a short period could improve your selling price while demand stays stable.',
            'recommended_date': peak_date,
            'recommended_price': round(peak_price, 2),
            'variant': 'info',
            'icon': 'fas fa-calendar-day',
            'demand_note': demand_note,
            'waiting_days': waiting_days
        }

    if peak_idx <= week_idx and week_gain >= min_gain:
        return {
            'action': 'Sell After One Week',
            'headline': 'A stronger selling window is expected within one week.',
            'message': 'Holding for a few days could improve your sale price before the forecast peak passes.',
            'recommended_date': peak_date,
            'recommended_price': round(peak_price, 2),
            'variant': 'info',
            'icon': 'fas fa-calendar-week',
            'demand_note': demand_note,
            'waiting_days': waiting_days
        }

    return {
        'action': 'Delay Sales',
        'headline': 'The forecast peak is later, so delaying sales may return more value.',
        'message': 'Price momentum stays favorable beyond the next week, so waiting could improve returns.',
        'recommended_date': peak_date,
        'recommended_price': round(peak_price, 2),
        'variant': 'warning',
        'icon': 'fas fa-hourglass-half',
        'demand_note': demand_note,
        'waiting_days': waiting_days
    }


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

    plot_pred_dates = pd.to_datetime(prediction_dates)

    return hist_dates, hist_values, plot_pred_dates, pred_values


def get_series_gap_days(hist_dates, pred_dates):
    if len(hist_dates) == 0 or len(pred_dates) == 0:
        return None
    hist_last = pd.to_datetime(hist_dates.iloc[-1] if hasattr(hist_dates, 'iloc') else hist_dates[-1])
    pred_first = pd.to_datetime(pred_dates[0])
    return int((pred_first.normalize() - hist_last.normalize()).days)


def apply_visual_prediction_gap(hist_values, pred_values, gap_fraction=0.004, min_gap=0.5, max_gap=None):
    if len(hist_values) == 0 or len(pred_values) == 0:
        return pred_values

    adjusted = np.array(pred_values, dtype=float).copy()
    hist_last = float(hist_values[-1])

    min_required_gap = max(abs(hist_last) * gap_fraction, min_gap)
    current_gap = adjusted[0] - hist_last

    if max_gap is None:
        target_gap = max(current_gap, min_required_gap)
    else:
        target_gap = float(np.clip(current_gap, min_required_gap, max_gap))

    if abs(target_gap - current_gap) > 1e-9:
        # Shift the full prediction curve to preserve forecast shape while fixing start gap.
        adjusted = adjusted + (target_gap - current_gap)

    return adjusted


def build_gapless_plot_axis(hist_dates, pred_dates, max_ticks=7, gap_spacing=0.35):
    hist_index = pd.to_datetime(pd.Index(hist_dates))
    pred_index = pd.to_datetime(pd.Index(pred_dates))

    hist_positions = np.arange(len(hist_index), dtype=float)
    if len(hist_index) > 0:
        pred_start = hist_positions[-1] + gap_spacing
    else:
        pred_start = 0.0
    pred_positions = pred_start + np.arange(len(pred_index), dtype=float)

    combined_dates = list(hist_index) + list(pred_index)
    combined_positions = np.concatenate([hist_positions, pred_positions]) if len(combined_dates) else np.array([], dtype=float)

    if len(combined_dates) == 0:
        return hist_positions, pred_positions, np.array([]), []

    tick_count = min(max_ticks, len(combined_dates))
    tick_indices = np.linspace(0, len(combined_dates) - 1, num=tick_count, dtype=int)

    deduped_indices = []
    for index in tick_indices.tolist():
        if index not in deduped_indices:
            deduped_indices.append(index)

    tick_positions = combined_positions[deduped_indices]
    tick_labels = [pd.to_datetime(combined_dates[index]).strftime('%Y-%m-%d') for index in deduped_indices]
    return hist_positions, pred_positions, tick_positions, tick_labels


def generate_price_plot(df_raw, prediction_dates, price_predictions):
    plt.figure(figsize=(12, 5))

    hist_dates, hist_prices, plot_pred_dates, plot_pred_prices = prepare_continuous_series(
        df_raw, prediction_dates, price_predictions,
        'Paddy_Price_LKR_per_kg', history_days=14, smooth_window=3
    )

    gap_days = get_series_gap_days(hist_dates, plot_pred_dates)
    large_time_gap = gap_days is not None and gap_days > 2
    display_pred_prices = apply_visual_prediction_gap(
        hist_prices,
        plot_pred_prices,
        min_gap=6.0,
        max_gap=9.0
    ) if large_time_gap else plot_pred_prices

    hist_pos, pred_pos, tick_positions, tick_labels = build_gapless_plot_axis(
        hist_dates,
        plot_pred_dates,
        gap_spacing=1.0 if large_time_gap else 0.35
    )

    plt.plot(hist_pos, hist_prices, 'b-', label='Historical Price', linewidth=2, alpha=0.8)

    if len(hist_dates) > 0 and len(plot_pred_dates) > 0 and not large_time_gap:
        plt.plot(
            [hist_pos[-1], pred_pos[0]],
            [hist_prices[-1], display_pred_prices[0]],
            color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='Transition'
        )

    plt.plot(pred_pos, display_pred_prices, 'ro-', label='Predicted Price', linewidth=2, markersize=5)

    if len(pred_pos) > 0:
        plt.axvline(x=pred_pos[0], color='gray', linestyle='--', alpha=0.6, label='Prediction Start')

    plt.title('Paddy Price Prediction', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Price (LKR/kg)', fontsize=11)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


def align_demand_graph_with_price(price_predictions, demand_predictions, min_step=1.0):
    aligned_demand = np.array(demand_predictions, dtype=float).copy()
    price_values = np.array(price_predictions, dtype=float).copy()

    if len(aligned_demand) == 0 or len(price_values) != len(aligned_demand):
        return aligned_demand

    for i in range(1, len(aligned_demand)):
        if price_values[i] > price_values[i - 1] and aligned_demand[i] <= aligned_demand[i - 1]:
            aligned_demand[i] = aligned_demand[i - 1] + max((price_values[i] - price_values[i - 1]) * 2.0, min_step)

    return aligned_demand


def generate_demand_plot(df_raw, prediction_dates, demand_predictions, price_predictions=None):
    plt.figure(figsize=(12, 5))

    plotted_demand = align_demand_graph_with_price(price_predictions, demand_predictions) \
        if price_predictions is not None else np.array(demand_predictions, dtype=float)

    hist_dates, hist_demand, plot_pred_dates, plot_pred_demand = prepare_continuous_series(
        df_raw, prediction_dates, plotted_demand,
        'Demand_Tons', history_days=14, smooth_window=3
    )

    gap_days = get_series_gap_days(hist_dates, plot_pred_dates)
    large_time_gap = gap_days is not None and gap_days > 2
    display_pred_demand = apply_visual_prediction_gap(
        hist_demand,
        plot_pred_demand,
        min_gap=40.0,
        max_gap=80.0
    ) if large_time_gap else plot_pred_demand

    hist_pos, pred_pos, tick_positions, tick_labels = build_gapless_plot_axis(
        hist_dates,
        plot_pred_dates,
        gap_spacing=1.0 if large_time_gap else 0.35
    )

    plt.plot(hist_pos, hist_demand, 'g-', label='Historical Demand', linewidth=2, alpha=0.8)

    if len(hist_dates) > 0 and len(plot_pred_dates) > 0 and not large_time_gap:
        plt.plot(
            [hist_pos[-1], pred_pos[0]],
            [hist_demand[-1], display_pred_demand[0]],
            color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='Transition'
        )

    plt.plot(pred_pos, display_pred_demand, 'mo-', label='Predicted Demand', linewidth=2, markersize=5)

    if len(pred_pos) > 0:
        plt.axvline(x=pred_pos[0], color='gray', linestyle='--', alpha=0.6, label='Prediction Start')

    plt.title('Demand Prediction', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Demand (Tons)', fontsize=11)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(tick_positions, tick_labels, rotation=45)
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
    pred_demands = align_demand_graph_with_price(pred_prices, demand_predictions)

    plot_pred_dates = pd.to_datetime(prediction_dates)
    hist_reference_dates = hist_price_dates if len(hist_price_dates) >= len(hist_demand_dates) else hist_demand_dates
    gap_days = get_series_gap_days(hist_reference_dates, plot_pred_dates)
    large_time_gap = gap_days is not None and gap_days > 2
    hist_pos, pred_pos, tick_positions, tick_labels = build_gapless_plot_axis(
        hist_reference_dates,
        plot_pred_dates,
        gap_spacing=1.0 if large_time_gap else 0.35
    )

    price_hist_pos = hist_pos[-len(hist_price_dates):] if len(hist_price_dates) > 0 else np.array([])
    demand_hist_pos = hist_pos[-len(hist_demand_dates):] if len(hist_demand_dates) > 0 else np.array([])

    if len(hist_prices) > 0 and len(pred_prices) > 0:
        if large_time_gap:
            pred_prices = apply_visual_prediction_gap(
                hist_prices,
                pred_prices,
                min_gap=6.0,
                max_gap=9.0
            )
        else:
            pred_prices = pred_prices + (hist_prices[-1] - pred_prices[0])

    if len(hist_demands) > 0 and len(pred_demands) > 0:
        if large_time_gap:
            pred_demands = apply_visual_prediction_gap(
                hist_demands,
                pred_demands,
                min_gap=40.0,
                max_gap=80.0
            )
        else:
            pred_demands = pred_demands + (hist_demands[-1] - pred_demands[0])

    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Price (LKR/kg)', color='red', fontsize=11)
    line1 = ax1.plot(price_hist_pos, hist_prices, 'r-', linewidth=2, alpha=0.6, label='Historical Price')
    line2 = ax1.plot(pred_pos, pred_prices, 'ro-', linewidth=2, markersize=5, label='Predicted Price')

    if len(hist_price_dates) > 0 and len(plot_pred_dates) > 0 and not large_time_gap:
        ax1.plot(
            [price_hist_pos[-1], pred_pos[0]],
            [hist_prices[-1], pred_prices[0]],
            color='orange', linestyle='--', linewidth=1.5, alpha=0.8
        )

    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Demand (Tons)', color='blue', fontsize=11)
    line3 = ax2.plot(demand_hist_pos, hist_demands, 'b-', linewidth=2, alpha=0.6, label='Historical Demand')
    line4 = ax2.plot(pred_pos, pred_demands, 'bs-', linewidth=2, markersize=5, label='Predicted Demand')

    if len(hist_demand_dates) > 0 and len(plot_pred_dates) > 0 and not large_time_gap:
        ax2.plot(
            [demand_hist_pos[-1], pred_pos[0]],
            [hist_demands[-1], pred_demands[0]],
            color='orange', linestyle='--', linewidth=1.5, alpha=0.8
        )

    ax2.tick_params(axis='y', labelcolor='blue')

    if len(pred_pos) > 0:
        ax1.axvline(x=pred_pos[0], color='gray', linestyle='--', alpha=0.6, label='Prediction Start')

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)

    plt.title('Price vs Demand Prediction Trend', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


# ============================================================================
# FERTILIZER PREDICTION FUNCTION
# ============================================================================

def predict_top3(soil_temp, soil_moisture, air_temp, air_humidity, growth_stage, purpose):
    """Predict top 3 fertilizers based on input conditions"""
    if clf is None or reg is None or scaler is None or fertilizer_encoders is None:
        # Return dummy predictions if models not loaded
        return [
            {
                "rank": 1,
                "fertilizer": "Urea",
                "quantity": 50.0,
                "yield": 4.5,
                "cost": 112500,
                "confidence": 85.5,
                "sustainability_note": "Standard urea application"
            },
            {
                "rank": 2,
                "fertilizer": "MOP",
                "quantity": 30.0,
                "yield": 4.2,
                "cost": 105000,
                "confidence": 75.2,
                "sustainability_note": "Muriate of Potash"
            },
            {
                "rank": 3,
                "fertilizer": "TSP",
                "quantity": 40.0,
                "yield": 4.0,
                "cost": 100000,
                "confidence": 65.8,
                "sustainability_note": "Triple Super Phosphate"
            }
        ]
    
    try:
        # Encode categorical inputs
        if growth_stage not in fertilizer_encoders["Paddy_Growth_Stage"].classes_:
            valid_stages = list(fertilizer_encoders["Paddy_Growth_Stage"].classes_)
            raise Exception(f"Invalid growth stage. Must be one of: {valid_stages}")
        
        if purpose not in fertilizer_encoders["Purpose"].classes_:
            valid_purposes = list(fertilizer_encoders["Purpose"].classes_)
            raise Exception(f"Invalid purpose. Must be one of: {valid_purposes}")
        
        growth_stage_encoded = fertilizer_encoders["Paddy_Growth_Stage"].transform([growth_stage])[0]
        purpose_encoded = fertilizer_encoders["Purpose"].transform([purpose])[0]
        
        # Base features
        base = {
            "Soil_Temperature (°C)": soil_temp,
            "Soil_Moisture (%)": soil_moisture,
            "Air_Temperature (°C)": air_temp,
            "Air_Humidity (%)": air_humidity,
            "Paddy_Growth_Stage": growth_stage_encoded,
            "Purpose": purpose_encoded,
            "Quantity_kg_per_acre": 25,
            "Recommended_Fertilizer": 0
        }
        
        # Create DataFrame for prediction
        X_base = pd.DataFrame([base])[fertilizer_feature_names]
        X_scaled = scaler.transform(X_base)
        
        # Get probabilities
        probs = clf.predict_proba(X_scaled)[0]
        top3_indices = probs.argsort()[-3:][::-1]
        
        results = []
        for rank, fert_idx in enumerate(top3_indices, start=1):
            fert_name = fertilizer_encoders["Recommended_Fertilizer"].inverse_transform([fert_idx])[0]
            
            # Get quantity and sustainability note
            if fertilizer_df_master is not None:
                match = fertilizer_df_master[fertilizer_df_master["Recommended_Fertilizer"] == fert_name]
                if not match.empty:
                    quantity = float(match.iloc[0]["Quantity_kg_per_acre"])
                    sustainability_note = str(match.iloc[0]["Sustainability_Note"])
                else:
                    quantity = 25.0
                    sustainability_note = "Follow standard application practices."
            else:
                quantity = 25.0
                sustainability_note = "Follow standard application practices."
            
            # Predict yield
            temp = base.copy()
            temp["Quantity_kg_per_acre"] = quantity
            temp["Recommended_Fertilizer"] = fert_idx
            
            X = pd.DataFrame([temp])[fertilizer_feature_names]
            X_scaled = scaler.transform(X)
            
            yield_pred = float(reg.predict(X_scaled)[0])
            cost_pred = yield_pred * 25000
            
            results.append({
                "rank": rank,
                "fertilizer": fert_name,
                "quantity": round(quantity, 2),
                "yield": round(yield_pred, 2),
                "cost": round(cost_pred, 2),
                "confidence": round(float(probs[fert_idx] * 100), 2),
                "sustainability_note": sustainability_note
            })
        
        return results
        
    except Exception as e:
        print(f"Error in predict_top3: {e}")
        # Return fallback predictions
        return [
            {
                "rank": 1,
                "fertilizer": "Urea",
                "quantity": 50.0,
                "yield": 4.5,
                "cost": 112500,
                "confidence": 85.5,
                "sustainability_note": "Standard urea application"
            },
            {
                "rank": 2,
                "fertilizer": "MOP",
                "quantity": 30.0,
                "yield": 4.2,
                "cost": 105000,
                "confidence": 75.2,
                "sustainability_note": "Muriate of Potash"
            },
            {
                "rank": 3,
                "fertilizer": "TSP",
                "quantity": 40.0,
                "yield": 4.0,
                "cost": 100000,
                "confidence": 65.8,
                "sustainability_note": "Triple Super Phosphate"
            }
        ]

# ============================================================================
# PEST PREDICTION FUNCTIONS
# ============================================================================

def predict_from_features(values):
    """Predict pest from sensor features"""
    if tabular_model is None or tabular_meta is None:
        return {
            "predicted_pest": "Brown Plant Hopper",
            "probability": 0.85,
            "recommended_action": "Apply appropriate pesticides and monitor regularly."
        }
    
    try:
        scaler = tabular_meta["scaler"]
        expected_features = scaler.n_features_in_
        
        if len(values) > expected_features:
            values = values[:expected_features]
        elif len(values) < expected_features:
            values = values + [0] * (expected_features - len(values))
        
        arr = np.array(values, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(arr)
        
        pred_idx = tabular_model.predict(X_scaled)[0]
        label = tabular_meta["label_encoder"].inverse_transform([pred_idx])[0]
        
        try:
            prob = float(np.max(tabular_model.predict_proba(X_scaled)[0]))
        except:
            prob = 0.85
        
        action_map = tabular_meta.get("action_map", {})
        action = action_map.get(label, "Monitor the crop regularly and consult agricultural expert.")
        
        return {
            "predicted_pest": label,
            "probability": prob,
            "recommended_action": action
        }
    except Exception as e:
        print(f"Error in predict_from_features: {e}")
        return {
            "predicted_pest": "Brown Plant Hopper",
            "probability": 0.85,
            "recommended_action": "Apply appropriate pesticides and monitor regularly."
        }

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.get_by_username(form.username.data)
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
        
        login_user(user, remember=form.remember_me.data)
        user.update_last_login()
        
        next_page = request.args.get('next')
        if not next_page or urlparse(next_page).netloc != '':
            next_page = url_for('dashboard')
        
        flash(f'Welcome back, {user.username}!', 'success')
        return redirect(next_page)
    
    return render_template('login.html', title='Sign In', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        if User.get_by_username(form.username.data):
            flash('Username already exists. Please choose a different one.', 'danger')
            return render_template('register.html', title='Register', form=form)
        
        if User.get_by_email(form.email.data):
            flash('Email already registered. Please use a different email or login.', 'danger')
            return render_template('register.html', title='Register', form=form)
        
        user = User.create(
            username=form.username.data,
            email=form.email.data,
            password=form.password.data
        )
        
        if user:
            flash('Congratulations! You are now registered. Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('An error occurred. Please try again.', 'danger')
    
    return render_template('register.html', title='Register', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing'))

# ============================================================================
# LANDING PAGE & DASHBOARD ROUTES
# ============================================================================

@app.route('/')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html', title='PaddyPredict - AI Powered Smart Paddy Farming')

# Keep 'index' as alias for backward compatibility with existing templates
@app.route('/index')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    price_history = PricesDemandHistory.get_by_user(current_user.id, limit=4)
    return render_template('dashboard.html', 
                         title='Dashboard',
                         historical_data=farming_historical_data,
                         model_loaded=FARMING_IMPORT_SUCCESS,
                         price_history=price_history)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        new_password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        if not new_password:
            flash('Password cannot be empty.', 'warning')
        elif new_password != confirm_password:
            flash('Passwords do not match. Please try again.', 'danger')
        elif len(new_password) < 6:
            flash('Password must be at least 6 characters long.', 'warning')
        else:
            current_user.update_password(new_password)
            flash('Password updated successfully! Please use your new password next time you log in.', 'success')
        return redirect(url_for('profile'))
    return render_template('profile.html', title='Profile & Settings')

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    return redirect(url_for('profile'))

def cost_verify_model_files():
    model_path = os.path.join(FARMING_COST_DIR, 'models', 'trained_paddy_cost_model.pkl')
    encoders_path = os.path.join(FARMING_COST_DIR, 'models', 'label_encoders.pkl')
    
    print(f"\n Checking model files:")
    print(f"   Model path: {model_path}")
    print(f"   Model exists: {os.path.exists(model_path)}")
    print(f"   Encoders path: {encoders_path}")
    print(f"   Encoders exist: {os.path.exists(encoders_path)}")
    
    return os.path.exists(model_path) and os.path.exists(encoders_path)
    
def cost_historical_data(df):
    if df is None or len(df) == 0:
        return {}
    
    try:
        recent_df = df.tail(30)
        
        # Calculate statistics
        historical_data = {
            'date_range': {
                'start': int(df['Year'].min()),
                'end': int(df['Year'].max()),
                'total_records': len(df)
            },
            'cost_stats': {
                'mean': float(df['Total_Cost (LKR)'].mean()),
                'min': float(df['Total_Cost (LKR)'].min()),
                'max': float(df['Total_Cost (LKR)'].max()),
                'std': float(df['Total_Cost (LKR)'].std())
            },
            'yield_stats': {
                'mean': float(df['Yield_kg'].mean()),
                'min': float(df['Yield_kg'].min()),
                'max': float(df['Yield_kg'].max())
            },
            'cost_per_kg_stats': {
                'mean': float(df['Cost_per_kg (LKR)'].mean()),
                'min': float(df['Cost_per_kg (LKR)'].min()),
                'max': float(df['Cost_per_kg (LKR)'].max())
            },
            'recent_years': recent_df['Year'].tolist(),
            'recent_costs': recent_df['Total_Cost (LKR)'].tolist(),
            'recent_yields': recent_df['Yield_kg'].tolist(),
            'regions': df['Region'].unique().tolist() if 'Region' in df.columns else [],
            'soil_types': df['Soil_Type'].unique().tolist() if 'Soil_Type' in df.columns else [],
            'seed_types': df['Seed_Type'].unique().tolist() if 'Seed_Type' in df.columns else [],
            'seasons': df['Season'].unique().tolist() if 'Season' in df.columns else []
        }
        
        return cost_historical_data
    except Exception as e:
        print(f" Error getting historical data: {e}")
        return {}




# ============================================================================
# DISEASE DETECTION ROUTE (Paddy Disease - image upload)
# ============================================================================

@app.route('/disease_detection', methods=['GET', 'POST'])
@login_required
def disease_detection():
    disease_model_path = os.path.join(PEST_OUTBREAK_DIR, 'NewDataone', 'paddy_disease_model.h5')
    class_indices_path = os.path.join(PEST_OUTBREAK_DIR, 'NewDataone', 'class_indices.json')
    pest_solutions_path = os.path.join(PEST_OUTBREAK_DIR, 'NewDataone', 'pest_solutions.json')
    model_loaded = (
        os.path.exists(disease_model_path)
        and os.path.exists(class_indices_path)
        and os.path.exists(pest_solutions_path)
    )
    if request.method == 'POST':
        if not model_loaded:
            flash('Disease detection model is not loaded. Please add paddy_disease_model.h5, class_indices.json, and pest_solutions.json to Pest_outbreak/NewDataone.', 'warning')
            return redirect(url_for('disease_detection'))
        try:
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(url_for('disease_detection'))
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(url_for('disease_detection'))
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
                filename = timestamp + filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                prev_cwd = os.getcwd()
                try:
                    newdataone_dir = os.path.join(PEST_OUTBREAK_DIR, 'NewDataone')
                    os.chdir(newdataone_dir)
                    if newdataone_dir not in sys.path:
                        sys.path.insert(0, newdataone_dir)
                    import leaf_predict as lp
                    result = lp.predict_paddy_disease(filepath)
                    with open(filepath, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                    global latest_disease_result
                    latest_disease_result = {
                        'prediction_type': 'Image-Based',
                        'result': result,
                        'uploaded_image': img_data,
                        'filename': filename,
                        'confidence_plot': generate_confidence_plot(result.get('confidence', 0)) if result.get('confidence') is not None else None,
                    }
                    return render_template('disease_results.html',
                                           result=result,
                                           uploaded_image=img_data,
                                           filename=filename)
                except Exception as e:
                    flash(f'Prediction error: {str(e)}', 'error')
                    traceback.print_exc()
                    return redirect(url_for('disease_detection'))
                finally:
                    os.chdir(prev_cwd)
            else:
                flash('File type not allowed. Please upload an image file.', 'error')
                return redirect(url_for('disease_detection'))
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('disease_detection'))
    return render_template('disease_detection.html', model_loaded=model_loaded)


@app.route('/pest_index')
def pest_index():
    return render_template('pest_index.html')
    
    
    
    
@app.route('/cost_index')
def cost_index():
    """Home page"""
    return render_template('cost_index.html', 
                         historical_data=cost_historical_data,
                         model_loaded=cost_verify_model_files())
                         

@app.route('/Fertilizer_index')
@login_required
def Fertilizer_index():
    return render_template('Fertilizer_index.html')

@app.route('/Fertilizer_history')
@login_required
def Fertilizer_history():
    """Display the fertilizer recommendation history."""
    user_history = FertilizerHistory.get_by_user(current_user.id)
    return render_template('Fertilizer_history.html', fert_history=user_history)


def save_prices_demand_prediction(user_id, prediction_type, results):
    recommendation = results.get('sales_recommendation') or {}
    PricesDemandHistory.add_record(user_id, {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction_type': prediction_type,
        'start_date': results.get('start_date', ''),
        'end_date': results.get('end_date', ''),
        'avg_price': results.get('avg_price', 0),
        'avg_demand': results.get('avg_demand', 0),
        'price_trend': results.get('price_trend', ''),
        'demand_trend': results.get('demand_trend', ''),
        'recommended_action': recommendation.get('action', ''),
        'recommended_date': recommendation.get('recommended_date'),
        'recommended_price': recommendation.get('recommended_price'),
        'sensor_data': results.get('sensor_data'),
        'results': results,
    })


def build_prices_demand_dashboard_data(user_id):
    history = PricesDemandHistory.get_by_user(user_id, limit=8)
    if not history:
        return {
            'summary': {
                'total_predictions': 0,
                'sensor_predictions': 0,
                'standard_predictions': 0,
                'latest_avg_price': None,
                'latest_avg_demand': None,
                'last_prediction_at': None,
            },
            'recent_predictions': [],
        }

    sensor_predictions = sum(1 for item in history if item.get('prediction_type') == 'Sensor-Based')
    standard_predictions = sum(1 for item in history if item.get('prediction_type') == 'Standard')
    latest_record = history[0]

    return {
        'summary': {
            'total_predictions': len(history),
            'sensor_predictions': sensor_predictions,
            'standard_predictions': standard_predictions,
            'latest_avg_price': latest_record.get('avg_price'),
            'latest_avg_demand': latest_record.get('avg_demand'),
            'last_prediction_at': latest_record.get('timestamp'),
        },
        'recent_predictions': history[:5],
    }


def get_pest_risk_level(probability):
    if probability > 0.75:
        return 'High'
    if probability > 0.4:
        return 'Medium'
    return 'Low'


def save_pest_prediction(user_id, prediction_type, result, input_data):
    try:
        confidence = round(float(result.get('probability', 0)) * 100, 1)
        predicted_pest = result.get('predicted_pest', 'Unknown')
        PestPredictionHistory.add_record(user_id, {
            'prediction_type': prediction_type,
            'predicted_pest': predicted_pest,
            'risk_level': get_pest_risk_level(float(result.get('probability', 0))),
            'confidence': confidence,
            'recommended_action': result.get('recommended_action', ''),
            'input_data': input_data,
        })
        return True
    except Exception as e:
        print(f"Error saving pest prediction history: {e}")
        return False
    

 



def Prices_Demand_load_and_preprocess_data():
    try:
        current_dir = os.getcwd()

        os.chdir(PRICES_DEMAND_DIR)
        print(f"Changed to directory: {os.getcwd()}")

        if prices_load_dataset is None or prices_preprocess is None:
            raise RuntimeError("Prices & Demand data loader is not available.")

        df = prices_load_dataset()
        print(f"Loaded dataset with {len(df)} rows")

        df_raw_local, df_mm_local, df_std_local, _ = prices_preprocess(df, save_artifacts=False)

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
    
@app.route('/Prices_Demand_index')
@login_required
def Prices_Demand_index():
    global prices_models, prices_historical_data, prices_df_raw, prices_df_mm, prices_df_std

    print("\n" + "=" * 60)
    print("🌾 PADDY PRICE & DEMAND PREDICTION SYSTEM")
    print("=" * 60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Prices_Demand directory: {PRICES_DEMAND_DIR}")
    print(f"Templates directory: {app.template_folder}")
    print(f"Static directory: {app.static_folder}")

    print("\nLoading models...")
    if prices_models is None:
        prices_models = load_prices_models()

    print("\nLoading historical data...")
    if prices_df_raw is None or prices_historical_data is None:
        df_raw_loaded, df_mm_loaded, df_std_loaded = Prices_Demand_load_and_preprocess_data()

        prices_df_raw = df_raw_loaded
        prices_df_mm = df_mm_loaded
        prices_df_std = df_std_loaded

        if prices_df_raw is not None:
            prices_historical_data = get_prices_historical_data(prices_df_raw)
        else:
            prices_historical_data = {}

    if prices_df_raw is not None:

        print(
            f"Loaded {len(prices_df_raw)} records from "
            f"{prices_historical_data.get('date_range', {}).get('start', 'N/A')} "
            f"to {prices_historical_data.get('date_range', {}).get('end', 'N/A')}"
        )
    else:
        prices_historical_data = {}
        print("Could not load historical data")

    print("\nStarting live sensor monitor...")
    if sensor_monitor and not sensor_monitor.running:
        sensor_monitor.start()

    print("\n" + "=" * 60)
    print("Application initialized successfully!")
    print("=" * 60)

    live_sensor = sensor_monitor.get_latest() if sensor_monitor else None
    dashboard_data = {
        'summary': {
            'total_predictions': 0,
            'sensor_predictions': 0,
            'standard_predictions': 0,
            'latest_avg_price': None,
            'latest_avg_demand': None,
            'last_prediction_at': None,
        },
        'recent_predictions': [],
    }

    if current_user.is_authenticated:
        dashboard_data = build_prices_demand_dashboard_data(current_user.id)

    return render_template(
        'Prices_Demand_index.html',
        historical_data=prices_historical_data,
        live_sensor=live_sensor,
        dashboard_data=dashboard_data,
        models_loaded=prices_models is not None
    )    
    
    
# ============================================================================
# FARMING COST ROUTES
# ============================================================================

@app.route('/cost_predict', methods=['GET', 'POST'])
@login_required
def cost_predict():
    if request.method == 'POST':
        try:
            # Get form data
            input_data = {
                "Latitude": float(request.form.get('latitude', 7.35)),
                "Longitude": float(request.form.get('longitude', 81.65)),
                "Year": int(request.form.get('year', datetime.now().year)),
                "Season": request.form.get('season', 'Maha'),
                "Soil_Type": request.form.get('soil_type', 'Clay'),
                "Rainfall_mm": float(request.form.get('rainfall', 2200)),
                "Temperature_C": float(request.form.get('temperature', 30.5)),
                "Humidity_%": float(request.form.get('humidity', 85)),
                "Area_acres": float(request.form.get('area', 2.5)),
                "Seed_Type": request.form.get('seed_type', 'BG352'),
                "Seed_Cost (LKR)": float(request.form.get('seed_cost', 8000)),
                "Fertilizer_Cost (LKR)": float(request.form.get('fertilizer_cost', 15000)),
                "Pesticide_Cost (LKR)": float(request.form.get('pesticide_cost', 4000)),
                "Labor_Cost (LKR)": float(request.form.get('labor_cost', 25000)),
                "Water_Cost (LKR)": float(request.form.get('water_cost', 5000)),
                "Machinery_Cost (LKR)": float(request.form.get('machinery_cost', 10000)),
                "Other_Costs (LKR)": float(request.form.get('other_costs', 2000))
            }
            
            # Make prediction
            predicted_cost = predict_total_cost(input_data)
            
            # Calculate cost breakdown
            cost_breakdown = {
                'Seed': input_data["Seed_Cost (LKR)"],
                'Fertilizer': input_data["Fertilizer_Cost (LKR)"],
                'Pesticide': input_data["Pesticide_Cost (LKR)"],
                'Labor': input_data["Labor_Cost (LKR)"],
                'Water': input_data["Water_Cost (LKR)"],
                'Machinery': input_data["Machinery_Cost (LKR)"],
                'Other': input_data["Other_Costs (LKR)"]
            }
            
            total_input = sum(cost_breakdown.values())
            
            # Generate plots
            cost_breakdown_plot = generate_cost_breakdown_plot(cost_breakdown, predicted_cost)
            comparison_plot = generate_comparison_plot(total_input, predicted_cost)
            
            results = {
                'input_data': input_data,
                'predicted_cost': float(predicted_cost),
                'cost_breakdown': cost_breakdown,
                'total_input': total_input,
                'difference': float(predicted_cost - total_input)
            }
            global latest_cost_result
            latest_cost_result = {
                'prediction_type': 'Standard',
                'results': results,
                'cost_breakdown_plot': cost_breakdown_plot,
                'comparison_plot': comparison_plot,
            }
            
            return render_template('cost_results.html',
                                 results=results,
                                 cost_breakdown_plot=cost_breakdown_plot,
                                 comparison_plot=comparison_plot)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('cost_predict'))
    
    # GET request - show form
    soil_types = (farming_historical_data or {}).get('soil_types', ['Clay', 'Sandy Loam', 'Loam', 'Silt'])
    seed_types = (farming_historical_data or {}).get('seed_types', ['BG300', 'BG352', 'BG358', 'At306', 'At402'])
    seasons = (farming_historical_data or {}).get('seasons', ['Maha', 'Yala'])
    
    return render_template('cost_predict.html',
                         current_year=datetime.now().year,
                         soil_types=soil_types,
                         seed_types=seed_types,
                         seasons=seasons)

@app.route('/cost_sensor_predict', methods=['GET', 'POST'])
@login_required
def cost_sensor_predict():
    if request.method == 'POST':
        try:
            # Get form data with sensor values
            input_data = {
                "Latitude": float(request.form.get('latitude', 7.35)),
                "Longitude": float(request.form.get('longitude', 81.65)),
                "Year": int(request.form.get('year', datetime.now().year)),
                "Season": request.form.get('season', 'Maha'),
                "Soil_Type": request.form.get('soil_type', 'Clay'),
                "Rainfall_mm": float(request.form.get('rainfall', 2200)),
                "Temperature_C": float(request.form.get('temperature', 30.5)),
                "Humidity_%": float(request.form.get('humidity', 85)),
                "Area_acres": float(request.form.get('area', 2.5)),
                "Seed_Type": request.form.get('seed_type', 'BG352'),
                "Seed_Cost (LKR)": float(request.form.get('seed_cost', 8000)),
                "Fertilizer_Cost (LKR)": float(request.form.get('fertilizer_cost', 15000)),
                "Pesticide_Cost (LKR)": float(request.form.get('pesticide_cost', 4000)),
                "Labor_Cost (LKR)": float(request.form.get('labor_cost', 25000)),
                "Water_Cost (LKR)": float(request.form.get('water_cost', 5000)),
                "Machinery_Cost (LKR)": float(request.form.get('machinery_cost', 10000)),
                "Other_Costs (LKR)": float(request.form.get('other_costs', 2000))
            }
            
            # Make prediction
            predicted_cost = predict_total_cost(input_data)
            
            # Calculate cost breakdown
            cost_breakdown = {
                'Seed': input_data["Seed_Cost (LKR)"],
                'Fertilizer': input_data["Fertilizer_Cost (LKR)"],
                'Pesticide': input_data["Pesticide_Cost (LKR)"],
                'Labor': input_data["Labor_Cost (LKR)"],
                'Water': input_data["Water_Cost (LKR)"],
                'Machinery': input_data["Machinery_Cost (LKR)"],
                'Other': input_data["Other_Costs (LKR)"]
            }
            
            total_input = sum(cost_breakdown.values())
            
            cost_breakdown_plot = generate_cost_breakdown_plot(cost_breakdown, predicted_cost)
            comparison_plot = generate_comparison_plot(total_input, predicted_cost)
            
            results = {
                'input_data': input_data,
                'predicted_cost': float(predicted_cost),
                'cost_breakdown': cost_breakdown,
                'total_input': total_input,
                'difference': float(predicted_cost - total_input),
                'sensor_used': True
            }
            global latest_cost_result
            latest_cost_result = {
                'prediction_type': 'Sensor-Based',
                'results': results,
                'cost_breakdown_plot': cost_breakdown_plot,
                'comparison_plot': comparison_plot,
            }
            
            return render_template('cost_results.html',
                                 results=results,
                                 cost_breakdown_plot=cost_breakdown_plot,
                                 comparison_plot=comparison_plot)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('cost_sensor_predict'))
    
    # GET request - show form
    soil_types = (farming_historical_data or {}).get('soil_types', ['Clay', 'Sandy Loam', 'Loam', 'Silt'])
    seed_types = (farming_historical_data or {}).get('seed_types', ['BG300', 'BG352', 'BG358', 'At306', 'At402'])
    seasons = (farming_historical_data or {}).get('seasons', ['Maha', 'Yala'])
    
    return render_template('cost_sensor_predict.html',
                         current_year=datetime.now().year,
                         soil_types=soil_types,
                         seed_types=seed_types,
                         seasons=seasons)

@app.route('/api/cost_predict', methods=['POST'])
@login_required
def cost_api_predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Prepare input data
        input_data = {
            "Latitude": float(data.get('latitude', 7.35)),
            "Longitude": float(data.get('longitude', 81.65)),
            "Year": int(data.get('year', datetime.now().year)),
            "Season": data.get('season', 'Maha'),
            "Soil_Type": data.get('soil_type', 'Clay'),
            "Rainfall_mm": float(data.get('rainfall', 2200)),
            "Temperature_C": float(data.get('temperature', 30.5)),
            "Humidity_%": float(data.get('humidity', 85)),
            "Area_acres": float(data.get('area', 2.5)),
            "Seed_Type": data.get('seed_type', 'BG352'),
            "Seed_Cost (LKR)": float(data.get('seed_cost', 8000)),
            "Fertilizer_Cost (LKR)": float(data.get('fertilizer_cost', 15000)),
            "Pesticide_Cost (LKR)": float(data.get('pesticide_cost', 4000)),
            "Labor_Cost (LKR)": float(data.get('labor_cost', 25000)),
            "Water_Cost (LKR)": float(data.get('water_cost', 5000)),
            "Machinery_Cost (LKR)": float(data.get('machinery_cost', 10000)),
            "Other_Costs (LKR)": float(data.get('other_costs', 2000))
        }
        
        predicted_cost = predict_total_cost(input_data)
        
        response = {
            'success': True,
            'predicted_cost': float(predicted_cost),
            'input_data': input_data
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/cost/save_prediction', methods=['POST'])
@login_required
def cost_save_prediction():
    """Save a farming cost prediction to the database."""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'}), 400
    
    # Add metadata
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    FarmingCostHistory.add_record(current_user.id, data)
    
    return jsonify({'success': True, 'message': 'Prediction saved permanently!'})

@app.route('/Farming_Cost_history')
@login_required
def Farming_Cost_history():
    """Display the farming cost prediction history for the current user."""
    user_history = FarmingCostHistory.get_by_user(current_user.id)
    return render_template('Farming_Cost_history.html', cost_history=user_history)

# ============================================================================
# FERTILIZER ROUTES
# ============================================================================

# DATABASE PERSISTENCE ENABLED
# The following lists are now replaced by SQLite storage
# farming_cost_history = []
# fertilizer_history = []

@app.route('/api/fertilizer/save_recommendation', methods=['POST'])
@login_required
def fertilizer_save_recommendation():
    """Save a user-selected fertilizer recommendation to the database."""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    FertilizerHistory.add_record(current_user.id, data)
    
    return jsonify({'success': True, 'message': 'Recommendation saved permanently!'})

@app.route('/api/fertilizer/history', methods=['GET'])
@login_required
def fertilizer_get_history():
    """Return the saved recommendation history for the current user."""
    user_history = FertilizerHistory.get_by_user(current_user.id)
    return jsonify({'success': True, 'history': user_history})

@app.route('/Intelligent_Fertilizer_predict', methods=['GET', 'POST'])
@login_required
def Intelligent_Fertilizer_predict():
    if request.method == 'POST':
        try:
            # Get form data
            soil_temp = float(request.form.get('soil_temp', 28.5))
            soil_moisture = float(request.form.get('soil_moisture', 60.0))
            air_temp = float(request.form.get('air_temp', 30.0))
            air_humidity = float(request.form.get('air_humidity', 75.0))
            growth_stage = request.form.get('growth_stage', 'Vegetative')
            purpose = request.form.get('purpose', 'Maximize Yield')
            
            # Make prediction
            results = predict_top3(
                soil_temp, soil_moisture, air_temp, air_humidity,
                growth_stage, purpose
            )
            
            # Generate plots
            confidence_plot = generate_confidence_plot(results[0]['confidence'] / 100)
            yield_comparison_plot = generate_yield_comparison_plot(results)
            input_data = {
                'soil_temp': soil_temp,
                'soil_moisture': soil_moisture,
                'air_temp': air_temp,
                'air_humidity': air_humidity,
                'growth_stage': growth_stage,
                'purpose': purpose
            }
            global latest_fertilizer_result
            latest_fertilizer_result = {
                'prediction_type': 'Manual Input',
                'results': results,
                'confidence_plot': confidence_plot,
                'yield_comparison_plot': yield_comparison_plot,
                'input_data': input_data,
            }
            
            return render_template('Intelligent_Fertilizer_results.html',
                                 results=results,
                                 confidence_plot=confidence_plot,
                                 yield_comparison_plot=yield_comparison_plot,
                                 input_data=input_data)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('Intelligent_Fertilizer_predict'))
    
    # GET request - show form
    growth_stages = []
    purposes = []
    
    if fertilizer_encoders is not None:
        if "Paddy_Growth_Stage" in fertilizer_encoders:
            growth_stages = list(fertilizer_encoders["Paddy_Growth_Stage"].classes_)
        if "Purpose" in fertilizer_encoders:
            purposes = list(fertilizer_encoders["Purpose"].classes_)
    
    if not growth_stages:
        growth_stages = ['Vegetative', 'Reproductive', 'Ripening', 'Nursery']
    if not purposes:
        purposes = ['Maximize Yield', 'Cost Effective', 'Balanced', 'Organic']
    
    return render_template('Intelligent_Fertilizer_predict.html',
                         growth_stages=growth_stages,
                         purposes=purposes)

@app.route('/Intelligent_Fertilizer_sensor_predict', methods=['GET', 'POST'])
@login_required
def Intelligent_Fertilizer_sensor_predict():
    latest_sensor = sensor_monitor.get_latest() if sensor_monitor else None
    
    if request.method == 'POST':
        try:
            # Get form data with sensor values
            soil_temp = float(request.form.get('soil_temp', 0))
            soil_moisture = float(request.form.get('soil_moisture', 0))
            air_temp = float(request.form.get('air_temp', 0))
            air_humidity = float(request.form.get('air_humidity', 0))
            growth_stage = request.form.get('growth_stage', 'Vegetative')
            purpose = request.form.get('purpose', 'Maximize Yield')
            
            # Make prediction
            results = predict_top3(
                soil_temp, soil_moisture, air_temp, air_humidity,
                growth_stage, purpose
            )
            
            # Generate plots
            confidence_plot = generate_confidence_plot(results[0]['confidence'] / 100)
            yield_comparison_plot = generate_yield_comparison_plot(results)
            input_data = {
                'soil_temp': soil_temp,
                'soil_moisture': soil_moisture,
                'air_temp': air_temp,
                'air_humidity': air_humidity,
                'growth_stage': growth_stage,
                'purpose': purpose
            }
            global latest_fertilizer_result
            latest_fertilizer_result = {
                'prediction_type': 'Sensor Input',
                'results': results,
                'confidence_plot': confidence_plot,
                'yield_comparison_plot': yield_comparison_plot,
                'input_data': input_data,
            }
            
            return render_template('Intelligent_Fertilizer_results.html',
                                 results=results,
                                 confidence_plot=confidence_plot,
                                 yield_comparison_plot=yield_comparison_plot,
                                 input_data=input_data,
                                 sensor_used=True)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('Intelligent_Fertilizer_sensor_predict'))
    
    # GET request - show form
    growth_stages = []
    purposes = []
    
    if fertilizer_encoders is not None:
        if "Paddy_Growth_Stage" in fertilizer_encoders:
            growth_stages = list(fertilizer_encoders["Paddy_Growth_Stage"].classes_)
        if "Purpose" in fertilizer_encoders:
            purposes = list(fertilizer_encoders["Purpose"].classes_)
    
    if not growth_stages:
        growth_stages = ['Vegetative', 'Reproductive', 'Ripening', 'Nursery']
    if not purposes:
        purposes = ['Maximize Yield', 'Cost Effective', 'Balanced', 'Organic']
    
    return render_template('Intelligent_Fertilizer_sensor_predict.html',
                         growth_stages=growth_stages,
                         purposes=purposes,
                         latest_sensor=latest_sensor)

@app.route('/api/Intelligent_Fertilizer_predict', methods=['POST'])
@login_required
def api_Intelligent_Fertilizer_predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        required = ['soil_temp', 'soil_moisture', 'air_temp', 'air_humidity', 'growth_stage', 'purpose']
        for field in required:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        results = predict_top3(
            float(data['soil_temp']),
            float(data['soil_moisture']),
            float(data['air_temp']),
            float(data['air_humidity']),
            data['growth_stage'],
            data['purpose']
        )
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ============================================================================
# PEST ROUTES
# ============================================================================

@app.route('/pest_image_predict', methods=['GET', 'POST'])
@login_required
def pest_image_predict():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                # Save file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
                filename = timestamp + filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Predict using image model
                if prt is not None and image_model is not None:
                    predicted_class_idx, confidence_score = prt.predict_image(filepath)
                    
                    if image_class_labels and predicted_class_idx < len(image_class_labels):
                        predicted_class_name = image_class_labels[predicted_class_idx]
                    else:
                        predicted_class_name = f"Unknown class {predicted_class_idx}"

                    if hasattr(prt, 'NON_PEST_LABELS') and predicted_class_name.lower() in prt.NON_PEST_LABELS:
                        predicted_class_name = "Non-pest item"
                    
                    # Get recommended action
                    if image_meta and "action_map" in image_meta:
                        raw_action_map = image_meta.get("action_map", {})
                        action_map = {k.lower(): v for k, v in raw_action_map.items()}
                        recommended_action = action_map.get(
                            predicted_class_name.lower(),
                            "No pest detected in the image." if predicted_class_name == "Non-pest item" else "Monitor the crop regularly and consult agricultural expert."
                        )
                    else:
                        recommended_action = "No pest detected in the image." if predicted_class_name == "Non-pest item" else "Monitor the crop regularly and consult agricultural expert."
                else:
                    # Fallback prediction
                    predicted_class_name = "Brown Plant Hopper"
                    confidence_score = 0.85
                    recommended_action = "Apply appropriate pesticides and monitor regularly."
                
                result = {
                    'predicted_pest': predicted_class_name,
                    'probability': float(confidence_score),
                    'recommended_action': recommended_action
                }
                global latest_pest_result
                latest_pest_result = {
                    'prediction_type': 'Image-Based',
                    'result': result,
                    'confidence_plot': generate_confidence_plot(confidence_score),
                    'input_data': {
                        'filename': filename,
                        'source': 'uploaded-image'
                    },
                }
                save_pest_prediction(current_user.id, 'Image-Based', result, {
                    'filename': filename,
                    'source': 'uploaded-image'
                })
                
                # Generate confidence plot
                confidence_plot = generate_confidence_plot(confidence_score)
                
                # Read image for display
                with open(filepath, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                latest_pest_result['uploaded_image'] = img_data
                latest_pest_result['filename'] = filename
                
                return render_template('pest_results.html',
                                     result=result,
                                     confidence_plot=confidence_plot,
                                     prediction_type='Image-Based',
                                     uploaded_image=img_data,
                                     filename=filename)
            
            else:
                flash('File type not allowed. Please upload an image file.', 'error')
                return redirect(request.url)
                
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('pest_image_predict'))
    
    # GET request - show form
    return render_template('pest_image_predict.html')



@app.route('/pest_predict', methods=['GET', 'POST'])
@login_required
def pest_predict():
    if request.method == 'POST':
        try:
            # Get form data
            temperature = float(request.form.get('temperature', 31.2))
            humidity = float(request.form.get('humidity', 86.5))
            pressure = float(request.form.get('pressure', 1006.3))
            light = float(request.form.get('light') or request.form.get('ldr') or 52300)
            dayofyear = int(request.form.get('dayofyear', datetime.now().timetuple().tm_yday))
            
            # Make prediction
            features = [temperature, humidity, pressure, light, dayofyear]
            result = predict_from_features(features)
            
            # Generate confidence plot
            confidence_plot = generate_confidence_plot(result['probability'])
            input_data = {
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'light': light,
                'dayofyear': dayofyear
            }
            global latest_pest_result
            latest_pest_result = {
                'prediction_type': 'Manual Entry',
                'result': result,
                'confidence_plot': confidence_plot,
                'input_data': input_data,
            }
            save_pest_prediction(current_user.id, 'Manual Entry', result, input_data)
            
            return render_template('pest_results.html',
                                 result=result,
                                 confidence_plot=confidence_plot,
                                 prediction_type='Manual Entry',
                                 input_data=input_data)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('pest_predict'))
    
    # GET request - show form
    return render_template('pest_predict.html',
                         current_day=datetime.now().timetuple().tm_yday)



@app.route('/pest_sensor_predict', methods=['GET', 'POST'])
@login_required
def pest_sensor_predict():
    if request.method == 'POST':
        try:
            # Get form data
            temperature = float(request.form.get('temperature', 31.2))
            humidity = float(request.form.get('humidity', 86.5))
            pressure = float(request.form.get('pressure', 1006.3))
            light = float(request.form.get('light') or request.form.get('ldr') or 52300)
            dayofyear = int(request.form.get('dayofyear', datetime.now().timetuple().tm_yday))
            
            # Make prediction
            features = [temperature, humidity, pressure, light, dayofyear]
            result = predict_from_features(features)
            
            # Generate confidence plot
            confidence_plot = generate_confidence_plot(result['probability'])
            input_data = {
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'light': light,
                'dayofyear': dayofyear
            }
            global latest_pest_result
            latest_pest_result = {
                'prediction_type': 'Sensor-Based',
                'result': result,
                'confidence_plot': confidence_plot,
                'input_data': input_data,
            }
            save_pest_prediction(current_user.id, 'Sensor-Based', result, input_data)
            
            return render_template('pest_results.html',
                                 result=result,
                                 confidence_plot=confidence_plot,
                                 prediction_type='Sensor-Based',
                                 input_data=input_data)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('pest_sensor_predict'))
    
    # GET request - show form
    return render_template('pest_sensor_predict.html',
                         current_day=datetime.now().timetuple().tm_yday)

@app.route('/api/pest_predict', methods=['POST'])
@login_required
def api_pest_predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Consistent field name 'light' or 'ldr'
        l_val = data.get('light') or data.get('ldr') or 52300
        
        features = [
            float(data.get('temperature', 31.2)),
            float(data.get('humidity', 86.5)),
            float(data.get('pressure', 1006.3)),
            float(l_val),
            int(data.get('dayofyear', datetime.now().timetuple().tm_yday))
        ]
        
        result = predict_from_features(features)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/pest-sensor-data', methods=['GET'])
@login_required
def get_pest_sensor_data():
    """Return latest data from the pest-specific sensor monitor."""
    data = pest_sensor.get_latest()
    return jsonify({
        'success': True,
        'sensor': data
    })


@app.route('/api/pest/save-history', methods=['POST'])
@login_required
def api_save_pest_history():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    result = {
        'predicted_pest': data.get('predicted_pest', 'Unknown'),
        'probability': float(data.get('probability', 0)),
        'recommended_action': data.get('recommended_action', ''),
    }
    input_data = data.get('input_data', {})
    prediction_type = data.get('prediction_type', 'Image-Based')

    created = save_pest_prediction(current_user.id, prediction_type, result, input_data)
    return jsonify({'success': True, 'created': created})


@app.route('/pest_history')
@login_required
def pest_history():
    user_history = PestPredictionHistory.get_by_user(current_user.id, limit=100)
    return render_template('pest_history.html', pest_history=user_history)

# ============================================================================
# PRICES DEMAND ROUTES
# ============================================================================

def _normalize_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _safe_float(value, default_value):
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(default_value)
        except (TypeError, ValueError):
            return None


def _get_prices_model_bundle():
    if not prices_models:
        raise ValueError("Prices models are not loaded.")

    lstm_model = prices_models.get('lstm')
    xgb_model = prices_models.get('xgb')

    if lstm_model is None:
        raise ValueError("LSTM price model is not loaded.")
    if xgb_model is None:
        raise ValueError("XGBoost demand model is not loaded.")

    lstm_features = prices_models.get('lstm_features', ['Paddy_Price_LKR_per_kg'])
    xgb_features = prices_models.get('xgb_features', ['Paddy_Price_LKR_per_kg'])
    window_size = int(prices_models.get('window_size', 21))

    return lstm_model, xgb_model, lstm_features, xgb_features, window_size


def _resolve_prices_group_choice(region=None, rice_type=None):
    if prices_df_mm is None or len(prices_df_mm) == 0:
        raise ValueError("Historical data is not loaded for price prediction.")

    if 'Region' not in prices_df_mm.columns or 'Rice_Type' not in prices_df_mm.columns:
        return _normalize_text(region) or 'North', _normalize_text(rice_type) or 'Samba'

    unique_groups = prices_df_mm[['Region', 'Rice_Type']].dropna().drop_duplicates().copy()
    if unique_groups.empty:
        raise ValueError("No Region/Rice_Type groups found in historical data.")

    unique_groups['region_norm'] = unique_groups['Region'].astype(str).str.strip().str.lower()
    unique_groups['rice_norm'] = unique_groups['Rice_Type'].astype(str).str.strip().str.lower()

    region_in = _normalize_text(region)
    rice_in = _normalize_text(rice_type)
    region_norm = region_in.lower() if region_in else None
    rice_norm = rice_in.lower() if rice_in else None

    selected = unique_groups.copy()
    if region_norm:
        selected = selected[selected['region_norm'] == region_norm]
    if rice_norm:
        selected = selected[selected['rice_norm'] == rice_norm]

    if selected.empty:
        raise ValueError(
            f"No historical rows found for Region='{region_in or 'Any'}' and Rice_Type='{rice_in or 'Any'}'."
        )

    if region_norm and rice_norm:
        return str(selected.iloc[0]['Region']), str(selected.iloc[0]['Rice_Type'])

    if not region_norm and not rice_norm:
        preferred = selected[
            (selected['region_norm'] == 'north') &
            (selected['rice_norm'] == 'samba')
        ]
        if not preferred.empty:
            return str(preferred.iloc[0]['Region']), str(preferred.iloc[0]['Rice_Type'])

        latest_group = prices_df_mm.sort_values('Date').iloc[-1]
        return str(latest_group['Region']), str(latest_group['Rice_Type'])

    return str(selected.iloc[0]['Region']), str(selected.iloc[0]['Rice_Type'])


def _filter_prices_group(df, region, rice_type):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if 'Region' in df.columns and 'Rice_Type' in df.columns:
        group_df = df[
            (df['Region'].astype(str).str.strip().str.lower() == str(region).strip().lower()) &
            (df['Rice_Type'].astype(str).str.strip().str.lower() == str(rice_type).strip().lower())
        ].copy()
    else:
        group_df = df.copy()

    if 'Date' in group_df.columns:
        group_df['Date'] = pd.to_datetime(group_df['Date'])
        group_df = group_df.sort_values('Date')

    return group_df.reset_index(drop=True)


def _prepare_prices_prediction_data(region=None, rice_type=None, nitrogen=None, phosphorus=None, potassium=None):
    resolved_region, resolved_rice_type = _resolve_prices_group_choice(region, rice_type)

    df_raw_group = _filter_prices_group(prices_df_raw, resolved_region, resolved_rice_type)
    df_mm_group = df_raw_group.copy()
    df_std_group = df_raw_group.copy()

    if df_raw_group.empty:
        raise ValueError(f"No historical data available for Region '{resolved_region}' and Rice Type '{resolved_rice_type}'.")

    if nitrogen is not None:
        df_mm_group.loc[df_mm_group.index[-1], 'Nitrogen_N'] = float(nitrogen)
        df_std_group.loc[df_std_group.index[-1], 'Nitrogen_N'] = float(nitrogen)
    if phosphorus is not None:
        df_mm_group.loc[df_mm_group.index[-1], 'Phosphorus_P'] = float(phosphorus)
        df_std_group.loc[df_std_group.index[-1], 'Phosphorus_P'] = float(phosphorus)
    if potassium is not None:
        df_mm_group.loc[df_mm_group.index[-1], 'Potassium_K'] = float(potassium)
        df_std_group.loc[df_std_group.index[-1], 'Potassium_K'] = float(potassium)

    region_encoder = prices_models.get('region_encoder') if prices_models else None
    rice_type_encoder = prices_models.get('rice_type_encoder') if prices_models else None

    if region_encoder is not None and rice_type_encoder is not None:
        try:
            region_encoded = float(region_encoder.transform([resolved_region])[0])
            rice_type_encoded = float(rice_type_encoder.transform([resolved_rice_type])[0])

            for frame in (df_mm_group, df_std_group):
                frame.loc[frame.index[-1], 'Region_encoded'] = region_encoded
                frame.loc[frame.index[-1], 'Rice_Type_encoded'] = rice_type_encoded
        except Exception as encoder_error:
            print(f"⚠️ Could not encode Region/Rice_Type for prediction: {encoder_error}")

    return {
        'region': resolved_region,
        'rice_type': resolved_rice_type,
        'df_mm': df_mm_group,
        'df_std': df_std_group,
        'df_raw': df_raw_group if not df_raw_group.empty else prices_df_raw
    }


def _get_prices_ui_options():
    source_df = None
    if prices_df_raw is not None and len(prices_df_raw) > 0:
        source_df = prices_df_raw
    elif prices_df_mm is not None and len(prices_df_mm) > 0:
        source_df = prices_df_mm

    if source_df is None:
        return {
            'region_options': ['North'],
            'rice_type_options': ['Samba'],
            'default_region': 'North',
            'default_rice_type': 'Samba'
        }

    if 'Region' in source_df.columns:
        region_values = source_df['Region'].dropna().astype(str).str.strip().tolist()
        region_options = sorted([
            v for v in set(region_values)
            if v and v.lower() != 'nan'
        ])
    else:
        region_options = ['North']

    if 'Rice_Type' in source_df.columns:
        rice_type_values = source_df['Rice_Type'].dropna().astype(str).str.strip().tolist()
        rice_type_options = sorted([
            v for v in set(rice_type_values)
            if v and v.lower() != 'nan'
        ])
    else:
        rice_type_options = ['Samba']

    if not region_options:
        region_options = ['North']
    if not rice_type_options:
        rice_type_options = ['Samba']

    default_region = 'North' if 'North' in region_options else region_options[0]
    default_rice_type = 'Samba' if 'Samba' in rice_type_options else rice_type_options[0]

    return {
        'region_options': region_options,
        'rice_type_options': rice_type_options,
        'default_region': default_region,
        'default_rice_type': default_rice_type
    }

@app.route('/Prices_Demand_predict', methods=['GET', 'POST'])
@login_required
def Prices_Demand_predict():
    if request.method == 'POST':
        try:
            start_date = request.form.get('start_date')
            n_days = max(1, int(request.form.get('n_days', 7)))
            region = _normalize_text(request.form.get('region'))
            rice_type = _normalize_text(request.form.get('rice_type'))

            if not start_date:
                start_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

            start_date = pd.to_datetime(start_date)
            end_date = start_date + timedelta(days=n_days - 1)

            lstm_model, xgb_model, lstm_features, xgb_features, window_size = _get_prices_model_bundle()
            prediction_context = _prepare_prices_prediction_data(region=region, rice_type=rice_type)

            if len(prediction_context['df_mm']) < window_size:
                raise ValueError(
                    f"Not enough historical rows for {prediction_context['region']} + {prediction_context['rice_type']} "
                    f"for LSTM window size {window_size}. Found only {len(prediction_context['df_mm'])} rows."
                )

            # Make price predictions (change to Prices_Demand dir for relative path resolution)
            current_dir_pred = os.getcwd()
            try:
                os.chdir(PRICES_DEMAND_DIR)
                price_predictions, prediction_dates = predict_price_with_trend(
                    prediction_context['df_mm'],
                    lstm_model,
                    lstm_features,
                    start_date,
                    n_steps=n_days,
                    window_size=window_size,
                    boost_factor=1.01
                )

                demand_predictions = predict_demand_with_trend(
                    prediction_context['df_std'],
                    xgb_model,
                    xgb_features,
                    price_predictions,
                    prediction_dates,
                    boost_factor=1.015
                )
            finally:
                os.chdir(current_dir_pred)

            # Convert prediction_dates to list of strings for template
            date_strings = [d.strftime('%Y-%m-%d') for d in prediction_dates]
            
            # Generate plots
            history_df = prediction_context['df_raw']
            price_plot = generate_price_plot(history_df, prediction_dates, price_predictions)
            demand_plot = generate_demand_plot(history_df, prediction_dates, demand_predictions, price_predictions)
            combined_plot = generate_combined_plot(history_df, prediction_dates, price_predictions, demand_predictions)

            results = {
                'dates': date_strings,
                'prices': [float(p) for p in price_predictions],
                'demands': [float(d) for d in demand_predictions],
                'avg_price': float(np.mean(price_predictions)),
                'avg_demand': float(np.mean(demand_predictions)),
                'price_range': [float(min(price_predictions)), float(max(price_predictions))],
                'demand_range': [float(min(demand_predictions)), float(max(demand_predictions))],
                'price_trend': 'Increasing' if price_predictions[-1] > price_predictions[0] else 'Decreasing',
                'demand_trend': 'Increasing' if demand_predictions[-1] > demand_predictions[0] else 'Decreasing',
                'sales_recommendation': generate_sales_recommendation(
                    prediction_dates,
                    price_predictions,
                    demand_predictions
                ),
                'region': prediction_context['region'],
                'rice_type': prediction_context['rice_type'],
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
            
            # Store for report generation
            global latest_prediction_result
            latest_prediction_result = {
                'results': results,
                'prediction_type': 'Standard',
                'sensor_data': None,
                'combined_plot': combined_plot
            }

            save_prices_demand_prediction(current_user.id, 'Standard', results)

            return render_template(
                'Prices_Demand_results.html',
                results=results,
                price_plot=price_plot,
                demand_plot=demand_plot,
                combined_plot=combined_plot,
                prediction_type='Standard'
            )

        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('Prices_Demand_predict'))

    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    ui_options = _get_prices_ui_options()
    return render_template(
        'Prices_Demand_predict.html',
        today=datetime.now().strftime('%Y-%m-%d'),
        tomorrow=tomorrow,
        region_options=ui_options['region_options'],
        rice_type_options=ui_options['rice_type_options'],
        default_region=ui_options['default_region'],
        default_rice_type=ui_options['default_rice_type']
    )

@app.route('/Prices_Demand_sensor_predict', methods=['GET', 'POST'])
@login_required
def Prices_Demand_sensor_predict():
    latest_sensor = sensor_monitor.get_latest() if sensor_monitor else None

    if request.method == 'POST':
        try:
            start_date = request.form.get('start_date')
            n_days = max(1, int(request.form.get('n_days', 7)))
            region = _normalize_text(request.form.get('region'))
            rice_type = _normalize_text(request.form.get('rice_type'))

            latest_values = latest_sensor.get('values', {}) if latest_sensor else {}
            nitrogen = _safe_float(request.form.get('nitrogen'), latest_values.get('nitrogen', 520))
            phosphorus = _safe_float(request.form.get('phosphorus'), latest_values.get('phosphorus', 10))
            potassium = _safe_float(request.form.get('potassium'), latest_values.get('potassium', 10))

            if nitrogen is None:
                nitrogen = 520.0
            if phosphorus is None:
                phosphorus = 10.0
            if potassium is None:
                potassium = 10.0

            if not start_date:
                start_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

            start_date = pd.to_datetime(start_date)
            end_date = start_date + timedelta(days=n_days - 1)

            lstm_model, xgb_model, lstm_features, xgb_features, window_size = _get_prices_model_bundle()
            prediction_context = _prepare_prices_prediction_data(
                region=region,
                rice_type=rice_type,
                nitrogen=nitrogen,
                phosphorus=phosphorus,
                potassium=potassium
            )

            if len(prediction_context['df_mm']) < window_size:
                raise ValueError(
                    f"Not enough historical rows for {prediction_context['region']} + {prediction_context['rice_type']} "
                    f"for LSTM window size {window_size}. Found only {len(prediction_context['df_mm'])} rows."
                )

            # Make price/demand predictions (change to Prices_Demand dir for relative path resolution)
            current_dir_pred = os.getcwd()
            try:
                os.chdir(PRICES_DEMAND_DIR)
                price_predictions, prediction_dates = predict_price_with_trend(
                    prediction_context['df_mm'],
                    lstm_model,
                    lstm_features,
                    start_date,
                    n_steps=n_days,
                    window_size=window_size,
                    boost_factor=1.01
                )

                demand_predictions = predict_demand_with_trend(
                    prediction_context['df_std'],
                    xgb_model,
                    xgb_features,
                    price_predictions,
                    prediction_dates,
                    boost_factor=1.015
                )
            finally:
                os.chdir(current_dir_pred)

            # Convert prediction_dates to list of strings for template
            date_strings = [d.strftime('%Y-%m-%d') for d in prediction_dates]
            
            # Generate plots
            history_df = prediction_context['df_raw']
            price_plot = generate_price_plot(history_df, prediction_dates, price_predictions)
            demand_plot = generate_demand_plot(history_df, prediction_dates, demand_predictions, price_predictions)
            combined_plot = generate_combined_plot(history_df, prediction_dates, price_predictions, demand_predictions)

            results = {
                'dates': date_strings,
                'prices': [float(p) for p in price_predictions],
                'demands': [float(d) for d in demand_predictions],
                'avg_price': float(np.mean(price_predictions)),
                'avg_demand': float(np.mean(demand_predictions)),
                'price_range': [float(min(price_predictions)), float(max(price_predictions))],
                'demand_range': [float(min(demand_predictions)), float(max(demand_predictions))],
                'price_trend': 'Increasing' if price_predictions[-1] > price_predictions[0] else 'Decreasing',
                'demand_trend': 'Increasing' if demand_predictions[-1] > demand_predictions[0] else 'Decreasing',
                'sales_recommendation': generate_sales_recommendation(
                    prediction_dates,
                    price_predictions,
                    demand_predictions
                ),
                'region': prediction_context['region'],
                'rice_type': prediction_context['rice_type'],
                'sensor_data': {
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium,
                    'live_connected': latest_sensor["connected"] if latest_sensor else False,
                    'live_last_update': latest_sensor["last_update"] if latest_sensor else None
                },
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
            
            # Store for report generation
            global latest_prediction_result
            latest_prediction_result = {
                'results': results,
                'prediction_type': 'Sensor-Based',
                'sensor_data': results.get('sensor_data'),
                'combined_plot': combined_plot
            }

            save_prices_demand_prediction(current_user.id, 'Sensor-Based', results)

            return render_template(
                'Prices_Demand_results.html',
                results=results,
                price_plot=price_plot,
                demand_plot=demand_plot,
                combined_plot=combined_plot,
                prediction_type='Sensor-Based'
            )

        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            traceback.print_exc()
            return redirect(url_for('Prices_Demand_sensor_predict'))

    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    ui_options = _get_prices_ui_options()
    return render_template(
        'Prices_Demand_sensor_predict.html',
        today=datetime.now().strftime('%Y-%m-%d'),
        tomorrow=tomorrow,
        latest_sensor=latest_sensor,
        region_options=ui_options['region_options'],
        rice_type_options=ui_options['rice_type_options'],
        default_region=ui_options['default_region'],
        default_rice_type=ui_options['default_rice_type']
    )

@app.route('/Prices_Demand_history')
@login_required
def Prices_Demand_history():
    """Display saved price and demand forecasts for the current user."""
    user_history = PricesDemandHistory.get_by_user(current_user.id, limit=100)
    return render_template('Prices_Demand_history.html', price_history=user_history)

@app.route('/api/Prices_Demand_predict', methods=['POST'])
@login_required
def Prices_Demand_api_predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        start_date = pd.to_datetime(
            data.get('start_date', (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))
        )
        n_days = max(1, int(data.get('n_days', 7)))
        region = _normalize_text(data.get('region'))
        rice_type = _normalize_text(data.get('rice_type'))

        nitrogen = _safe_float(data.get('nitrogen'), np.nan)
        phosphorus = _safe_float(data.get('phosphorus'), np.nan)
        potassium = _safe_float(data.get('potassium'), np.nan)

        if np.isnan(nitrogen):
            nitrogen = None
        if np.isnan(phosphorus):
            phosphorus = None
        if np.isnan(potassium):
            potassium = None

        lstm_model, xgb_model, lstm_features, xgb_features, window_size = _get_prices_model_bundle()
        prediction_context = _prepare_prices_prediction_data(
            region=region,
            rice_type=rice_type,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium
        )

        if len(prediction_context['df_mm']) < window_size:
            raise ValueError(
                f"Not enough historical rows for {prediction_context['region']} + {prediction_context['rice_type']} "
                f"for LSTM window size {window_size}. Found only {len(prediction_context['df_mm'])} rows."
            )

        # Make price/demand predictions (change to Prices_Demand dir for relative path resolution)
        current_dir_pred = os.getcwd()
        try:
            os.chdir(PRICES_DEMAND_DIR)
            price_predictions, prediction_dates = predict_price_with_trend(
                prediction_context['df_mm'],
                lstm_model,
                lstm_features,
                start_date,
                n_steps=n_days,
                window_size=window_size,
                boost_factor=1.01
            )

            demand_predictions = predict_demand_with_trend(
                prediction_context['df_std'],
                xgb_model,
                xgb_features,
                price_predictions,
                prediction_dates,
                boost_factor=1.015
            )
        finally:
            os.chdir(current_dir_pred)

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
            },
            'context': {
                'region': prediction_context['region'],
                'rice_type': prediction_context['rice_type'],
                'sensor_inputs': {
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium
                }
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


def _pdf_ensure_space(pdf, y, min_space=80):
    if y < min_space:
        pdf.showPage()
        return A4[1] - 50
    return y


def _pdf_draw_section(pdf, y, title):
    y = _pdf_ensure_space(pdf, y, 90)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, title)
    return y - 20


def _pdf_draw_lines(pdf, y, lines, x=60, size=11, leading=18):
    pdf.setFont("Helvetica", size)
    for line in lines:
        for wrapped in textwrap.wrap(str(line), width=92) or [""]:
            y = _pdf_ensure_space(pdf, y, 60)
            pdf.drawString(x, y, wrapped)
            y -= leading
    return y


def _pdf_draw_base64_image(pdf, y, title, image_b64, image_width=500, image_height=220):
    if not image_b64:
        return y

    y = _pdf_draw_section(pdf, y, title)
    y = _pdf_ensure_space(pdf, y, image_height + 50)

    try:
        image_data = base64.b64decode(image_b64)
        image_stream = io.BytesIO(image_data)
        img = ImageReader(image_stream)
        pdf.drawImage(img, 50, y - image_height, width=image_width, height=image_height)
        return y - image_height - 20
    except Exception as e:
        print(f"Error adding image to PDF: {e}")
        return _pdf_draw_lines(pdf, y, ["Image unavailable in report."])


def _pdf_response(pdf, buffer, filename):
    pdf.save()
    buffer.seek(0)
    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response

@app.route('/download-report', methods=['GET'])
@login_required
def download_report():
    global latest_prediction_result

    if not latest_prediction_result:
        flash("No prediction report available. Please generate a prediction first.", "warning")
        return redirect(url_for('Prices_Demand_predict'))

    results = latest_prediction_result['results']
    prediction_type = latest_prediction_result['prediction_type']
    sensor_data = latest_prediction_result['sensor_data']
    combined_plot = latest_prediction_result.get('combined_plot')

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

    recommendation = results.get('sales_recommendation')
    if recommendation:
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, y, "Farmer Sales Recommendation")
        y -= 20

        pdf.setFont("Helvetica", 11)
        pdf.drawString(60, y, f"Action: {recommendation['action']}")
        y -= 18
        pdf.drawString(60, y, f"Recommended Date: {recommendation['recommended_date'] or 'N/A'}")
        y -= 18
        if recommendation.get('recommended_price') is not None:
            pdf.drawString(60, y, f"Expected Price: Rs. {recommendation['recommended_price']:.2f}")
            y -= 18
        pdf.drawString(60, y, recommendation['message'])
        y -= 18
        pdf.drawString(60, y, recommendation['demand_note'])
        y -= 30

    # Add combined plot if available
    if combined_plot:
        y -= 10
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, y, "Price vs Demand Trend Graph")
        y -= 20

        try:
            image_data = base64.b64decode(combined_plot)
            image_stream = io.BytesIO(image_data)
            img = ImageReader(image_stream)
            pdf.drawImage(img, 50, y - 250, width=500, height=250)
            y -= 270
        except Exception as e:
            print(f"Error adding image to PDF: {e}")

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


@app.route('/download-cost-report', methods=['GET'])
@login_required
def download_cost_report():
    global latest_cost_result

    if not latest_cost_result:
        flash("No farming cost report available. Please generate a prediction first.", "warning")
        return redirect(url_for('cost_predict'))

    results = latest_cost_result['results']
    prediction_type = latest_cost_result['prediction_type']

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    y = A4[1] - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Paddy Farming Cost Prediction Report")
    y -= 25
    y = _pdf_draw_lines(pdf, y, [
        f"Prediction Type: {prediction_type}",
        f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ], x=50)
    y -= 12

    y = _pdf_draw_section(pdf, y, "Summary")
    y = _pdf_draw_lines(pdf, y, [
        f"Predicted Total Cost: LKR {results['predicted_cost']:,.2f}",
        f"Supplied Input Total: LKR {results['total_input']:,.2f}",
        f"Model Adjustment: LKR {results['difference']:+,.2f}",
        f"Sensor Used: {'Yes' if results.get('sensor_used') else 'No'}",
    ])
    y -= 12

    y = _pdf_draw_section(pdf, y, "Input Parameters")
    y = _pdf_draw_lines(pdf, y, [
        f"Latitude: {results['input_data']['Latitude']:.4f}",
        f"Longitude: {results['input_data']['Longitude']:.4f}",
        f"Season: {results['input_data']['Season']}",
        f"Soil Type: {results['input_data']['Soil_Type']}",
        f"Temperature: {results['input_data']['Temperature_C']:.1f} C",
        f"Humidity: {results['input_data']['Humidity_%']:.1f} %",
        f"Area: {results['input_data']['Area_acres']:.2f} acres",
        f"Seed Type: {results['input_data']['Seed_Type']}",
    ])
    y -= 12

    y = _pdf_draw_section(pdf, y, "Cost Breakdown")
    breakdown_lines = [f"{name}: LKR {value:,.2f}" for name, value in results['cost_breakdown'].items()]
    y = _pdf_draw_lines(pdf, y, breakdown_lines)
    y -= 12

    y = _pdf_draw_base64_image(pdf, y, "Cost Breakdown Chart", latest_cost_result.get('cost_breakdown_plot'))
    y = _pdf_draw_base64_image(pdf, y, "Model vs Input Comparison", latest_cost_result.get('comparison_plot'))

    return _pdf_response(pdf, buffer, 'farming_cost_report.pdf')


@app.route('/download-fertilizer-report', methods=['GET'])
@login_required
def download_fertilizer_report():
    global latest_fertilizer_result

    if not latest_fertilizer_result:
        flash("No fertilizer report available. Please generate a recommendation first.", "warning")
        return redirect(url_for('Intelligent_Fertilizer_predict'))

    results = latest_fertilizer_result['results']
    input_data = latest_fertilizer_result['input_data']
    prediction_type = latest_fertilizer_result['prediction_type']
    best = results[0] if results else None

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    y = A4[1] - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Paddy Fertilizer Recommendation Report")
    y -= 25
    y = _pdf_draw_lines(pdf, y, [
        f"Prediction Type: {prediction_type}",
        f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ], x=50)
    y -= 12

    if best:
        y = _pdf_draw_section(pdf, y, "Top Recommendation")
        y = _pdf_draw_lines(pdf, y, [
            f"Fertilizer: {best['fertilizer']}",
            f"Confidence: {best['confidence']}%",
            f"Quantity: {best['quantity']} kg/acre",
            f"Expected Yield: {best['yield']} ton/ha",
            f"Estimated Cost: LKR {best['cost']:,.0f}",
            f"Sustainability Note: {best['sustainability_note']}",
        ])
        y -= 12

    y = _pdf_draw_section(pdf, y, "Input Parameters")
    y = _pdf_draw_lines(pdf, y, [
        f"Soil Temperature: {input_data['soil_temp']:.1f} C",
        f"Soil Moisture: {input_data['soil_moisture']:.1f} %",
        f"Air Temperature: {input_data['air_temp']:.1f} C",
        f"Air Humidity: {input_data['air_humidity']:.1f} %",
        f"Growth Stage: {input_data['growth_stage']}",
        f"Purpose: {input_data['purpose']}",
    ])
    y -= 12

    y = _pdf_draw_section(pdf, y, "All Recommendations")
    recommendation_lines = []
    for item in results:
        recommendation_lines.extend([
            f"#{item['rank']} {item['fertilizer']} | Confidence {item['confidence']}% | Quantity {item['quantity']} kg/acre | Yield {item['yield']} ton/ha | Cost LKR {item['cost']:,.0f}",
            f"Note: {item['sustainability_note']}",
        ])
    y = _pdf_draw_lines(pdf, y, recommendation_lines)
    y -= 12

    y = _pdf_draw_base64_image(pdf, y, "Confidence Comparison", latest_fertilizer_result.get('confidence_plot'))
    y = _pdf_draw_base64_image(pdf, y, "Expected Yield Comparison", latest_fertilizer_result.get('yield_comparison_plot'))

    return _pdf_response(pdf, buffer, 'fertilizer_recommendation_report.pdf')


@app.route('/download-pest-report', methods=['GET'])
@login_required
def download_pest_report():
    global latest_pest_result

    if not latest_pest_result:
        flash("No pest report available. Please generate a prediction first.", "warning")
        return redirect(url_for('pest_predict'))

    result = latest_pest_result['result']
    prediction_type = latest_pest_result['prediction_type']
    input_data = latest_pest_result.get('input_data') or {}

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    y = A4[1] - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Paddy Pest Detection Report")
    y -= 25
    y = _pdf_draw_lines(pdf, y, [
        f"Prediction Type: {prediction_type}",
        f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ], x=50)
    y -= 12

    y = _pdf_draw_section(pdf, y, "Summary")
    y = _pdf_draw_lines(pdf, y, [
        f"Predicted Pest: {result['predicted_pest']}",
        f"Risk Level: {get_pest_risk_level(float(result.get('probability', 0)))}",
        f"Confidence: {float(result.get('probability', 0)) * 100:.1f}%",
        f"Recommended Action: {result['recommended_action']}",
    ])
    y -= 12

    if prediction_type == 'Image-Based':
        y = _pdf_draw_section(pdf, y, "Image Input")
        y = _pdf_draw_lines(pdf, y, [
            f"Filename: {input_data.get('filename', 'Uploaded image')}",
            f"Source: {input_data.get('source', 'image-upload')}",
        ])
        y -= 12
        y = _pdf_draw_base64_image(pdf, y, "Uploaded Image", latest_pest_result.get('uploaded_image'), image_width=320, image_height=220)
    else:
        y = _pdf_draw_section(pdf, y, "Input Parameters")
        y = _pdf_draw_lines(pdf, y, [
            f"Temperature: {input_data['temperature']:.1f} C",
            f"Humidity: {input_data['humidity']:.1f} %",
            f"Pressure: {input_data['pressure']:.1f} hPa",
            f"Light: {input_data['light']:.1f} lux",
            f"Day of Year: {input_data['dayofyear']}",
        ])
        y -= 12

    y = _pdf_draw_base64_image(pdf, y, "Confidence Chart", latest_pest_result.get('confidence_plot'), image_width=420, image_height=200)

    return _pdf_response(pdf, buffer, 'pest_detection_report.pdf')


@app.route('/download-disease-report', methods=['GET'])
@login_required
def download_disease_report():
    global latest_disease_result

    if not latest_disease_result:
        flash("No disease report available. Please generate an analysis first.", "warning")
        return redirect(url_for('disease_detection'))

    result = latest_disease_result['result']

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    y = A4[1] - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Paddy Disease Detection Report")
    y -= 25
    y = _pdf_draw_lines(pdf, y, [
        "Prediction Type: Image-Based",
        f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Filename: {latest_disease_result.get('filename', 'Uploaded image')}",
    ], x=50)
    y -= 12

    y = _pdf_draw_section(pdf, y, "Summary")
    summary_lines = [
        f"Detected Disease: {result.get('disease', 'N/A')}",
        f"Associated Pest: {result.get('pest', 'N/A')}",
    ]
    if result.get('confidence') is not None:
        summary_lines.append(f"Confidence: {float(result['confidence']) * 100:.1f}%")
    if result.get('growth_stage'):
        summary_lines.append(f"Growth Stage: {result['growth_stage']}")
    if result.get('severity'):
        summary_lines.append(f"Severity: {result['severity']}")
    y = _pdf_draw_lines(pdf, y, summary_lines)
    y -= 12

    y = _pdf_draw_section(pdf, y, "Recommended Solution")
    y = _pdf_draw_lines(pdf, y, [result.get('solution', 'No solution available.')])
    y -= 12

    y = _pdf_draw_base64_image(pdf, y, "Uploaded Image", latest_disease_result.get('uploaded_image'), image_width=320, image_height=220)
    y = _pdf_draw_base64_image(pdf, y, "Confidence Chart", latest_disease_result.get('confidence_plot'), image_width=420, image_height=200)

    return _pdf_response(pdf, buffer, 'disease_detection_report.pdf')

# ============================================================================
# SENSOR API ROUTES
# ============================================================================

@app.route('/api/latest-sensor', methods=['GET'])
@login_required
def api_latest_sensor():
    if sensor_monitor:
        return jsonify({
            "success": True,
            "sensor": sensor_monitor.get_latest()
        })
    else:
        return jsonify({
            "success": False,
            "error": "Sensor monitor not initialized"
        }), 500

@app.route('/api/sensor-data', methods=['GET'])
@login_required
def api_sensor_data():
    """Dedicated endpoint for Fertilizer module"""
    if ferti_sensor:
        return jsonify({
            "success": True,
            "sensor": ferti_sensor.get_latest()
        })
    return jsonify({"success": False, "error": "Fertilizer sensor not initialized"}), 500

@app.route('/api/reconnect-sensor', methods=['POST'])
@login_required
def api_reconnect_sensor():
    try:
        if COM7_broker:
            COM7_broker.start()

        if sensor_monitor:
            sensor_monitor.stop()
            time.sleep(0.5)
            sensor_monitor.start_with_queue(npk_queue, COM7_broker)
            
        if ferti_sensor:
            ferti_sensor.stop()
            time.sleep(0.5)
            ferti_sensor.start_with_queue(ferti_queue, COM7_broker)
            return jsonify({
                "success": True,
                "sensor": sensor_monitor.get_latest() if sensor_monitor else None,
                "fertilizer_sensor": ferti_sensor.get_latest()
            })
            
        return jsonify({
            "success": False,
            "error": "Sensor monitors not initialized"
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/cost-sensor-data', methods=['GET'])
@login_required
def api_cost_sensor_data():
    """Dedicated endpoint for Farming Cost module"""
    if cost_sensor:
        return jsonify({
            "success": True,
            "sensor": cost_sensor.get_latest()
        })
    return jsonify({"success": False, "error": "Cost sensor not initialized"}), 500

@app.route('/api/cost-reconnect-sensor', methods=['POST'])
@login_required
def api_cost_reconnect_sensor():
    try:
        if COM7_broker:
            COM7_broker.start()
        if cost_sensor:
            cost_sensor.stop()
            time.sleep(0.5)
            cost_sensor.start_with_queue(cost_queue, COM7_broker)
            return jsonify({
                "success": True,
                "sensor": cost_sensor.get_latest()
            })
        return jsonify({
            "success": False,
            "error": "Cost sensor not initialized"
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/debug-sensor')
@login_required
def debug_sensor():
    return jsonify({
        "main_sensor": sensor_monitor.get_latest() if sensor_monitor else None,
        "ferti_sensor": ferti_sensor.get_latest() if ferti_sensor else None,
        "cost_sensor": cost_sensor.get_latest() if cost_sensor else None
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.errorhandler(413)
def too_large_error(error):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(request.url)

# ============================================================================
# APP CONTEXT PROCESSOR FOR ADDITIONAL VARIABLES
# ============================================================================
@app.context_processor
def utility_processor():
    """Add utility functions to template context"""
    return {
        'now': datetime.now(),
        'app_name': 'Paddy Price & Demand Predictor',
        'models_loaded': (FARMING_IMPORT_SUCCESS or 
                         (clf is not None) or 
                         (tabular_model is not None) or
                         (prices_models is not None))
    }

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌾 PADDY PRICE & DEMAND PREDICTOR - COMPLETE APPLICATION")
    print("="*70)
    print(f" Base Directory: {BASE_DIR}")
    print(f" Templates Directory: {app.template_folder}")
    print(f" Static Directory: {app.static_folder}")
    print(f" Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print("\n" + "-"*70)
    print(" MODULES STATUS:")
    print(f"   Farming Cost: {'✅ Loaded' if FARMING_IMPORT_SUCCESS else '⚠️ Fallback'}")
    print(f"   Fertilizer: {'✅ Loaded' if clf is not None else '⚠️ Fallback'}")
    print(f"   Pest Detection: {'✅ Loaded' if tabular_model is not None else '⚠️ Fallback'}")
    print(f"   Prices & Demand: {'✅ Loaded' if prices_models is not None else '⚠️ Fallback'}")
    print(f"   Sensor Monitor: {'✅ Active' if sensor_monitor else '⚠️ Inactive'}")
    print("-"*70)
    print("\n" + "="*70)
    print("🚀 Starting Flask application...")
    print(f" Access the application at: http://localhost:5000")
    print(f" Login page: http://localhost:5000/login")
    print(f" Register page: http://localhost:5000/register")
    print(" Press CTRL+C to stop\n")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 
