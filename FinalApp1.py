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
import traceback
import json
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
app.config['SECRET_KEY'] = os.urandom(24).hex()
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
from models import User, init_db
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

# Fertilizer globals
clf = None
reg = None
scaler = None
fertilizer_encoders = None
fertilizer_feature_names = None
fertilizer_df_master = None

# Pest globals
tabular_model = None
tabular_meta = None
image_model = None
image_meta = None
image_class_labels = []
prt = None
Visual = None

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
# LIVE SENSOR MONITOR CLASS
# ============================================================================
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
        self.partial_values = {}

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
        with self.lock:
            self.latest["connected"] = False

    def parse_line(self, text):
        if not text:
            return None
        line = text.strip().lower()
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
                    
                    with self.lock:
                        self.latest["last_raw"] = str(raw)
                        self.latest["last_decoded"] = text
                        self.latest["connected"] = True
                    
                    if parsed:
                        self.partial_values.update(parsed)
                    
                    if all(k in self.partial_values for k in ["nitrogen", "phosphorus", "potassium"]):
                        self.commit_partial_values()
                    
                    if "----" in text or "npk values" in text.lower():
                        self.commit_partial_values()
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
sensor_monitor.start()

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
        train_path = os.path.join(PEST_OUTBREAK_DIR, 'PestDataset', 'train')
        
        if os.path.exists(model_path) and os.path.exists(meta_path):
            image_model = load_model(model_path)
            image_meta = joblib.load(meta_path)
            
            if os.path.exists(train_path):
                image_class_labels = sorted([
                    d for d in os.listdir(train_path)
                    if os.path.isdir(os.path.join(train_path, d))
                ])
                print(f"✅ Found {len(image_class_labels)} pest classes")
            else:
                if 'class_names' in image_meta:
                    image_class_labels = image_meta['class_names']
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
        
        # Load XGBoost model
        xgb_path = os.path.join(models_path, 'xgb_demand_model_best_optimized.joblib')
        if os.path.exists(xgb_path):
            models_dict['xgb'] = joblib.load(xgb_path)
            print("✅ XGBoost demand model loaded")
        else:
            print("⚠️ XGBoost model not found")
            models_dict['xgb'] = None
        
        # Load feature columns
        xgb_features_path = os.path.join(models_path, 'feature_columns_optimized.joblib')
        if os.path.exists(xgb_features_path):
            models_dict['xgb_features'] = joblib.load(xgb_features_path)
            print(f"✅ Loaded {len(models_dict['xgb_features'])} demand features")
        else:
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
        return {
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d') if 'Date' in df.columns else '2023-01-01',
                'end': df['Date'].max().strftime('%Y-%m-%d') if 'Date' in df.columns else '2024-12-31',
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
            'recent_demands': recent_df['Demand_Tons'].tolist() if 'Demand_Tons' in recent_df.columns else []
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
    return render_template('dashboard.html', 
                         title='Dashboard',
                         historical_data=farming_historical_data,
                         model_loaded=FARMING_IMPORT_SUCCESS)



 



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
def Fertilizer_index():
    return render_template('Fertilizer_index.html')
    

 



def Prices_Demand_load_and_preprocess_data():
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
    
@app.route('/Prices_Demand_index')
def Prices_Demand_index():
    global models, Prices_Demand_historical_data, df_raw, df_mm, df_std

    print("\n" + "=" * 60)
    print("🌾 PADDY PRICE & DEMAND PREDICTION SYSTEM")
    print("=" * 60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Prices_Demand directory: {PRICES_DEMAND_DIR}")
    print(f"Templates directory: {app.template_folder}")
    print(f"Static directory: {app.static_folder}")

    print("\nLoading models...")
 

    print("\nLoading historical data...")
    df_raw_loaded, df_mm_loaded, df_std_loaded = Prices_Demand_load_and_preprocess_data()

    df_raw = df_raw_loaded
    df_mm = df_mm_loaded
    df_std = df_std_loaded

    if df_raw is not None:
        Prices_Demand_historical_data = get_historical_data(df_raw)

        print(
            f"Loaded {len(df_raw)} records from "
            f"{Prices_Demand_historical_data.get('date_range', {}).get('start', 'N/A')} "
            f"to {Prices_Demand_historical_data.get('date_range', {}).get('end', 'N/A')}"
        )
    else:
        Prices_Demand_historical_data = {}
        print("Could not load historical data")

    print("\nStarting live sensor monitor...")
    sensor_monitor.start()

    print("\n" + "=" * 60)
    print("Application initialized successfully!")
    print("=" * 60)

    return render_template(
        'Prices_Demand_index.html',
        historical_data=Prices_Demand_historical_data
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

# ============================================================================
# FERTILIZER ROUTES
# ============================================================================

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
            
            return render_template('Intelligent_Fertilizer_results.html',
                                 results=results,
                                 confidence_plot=confidence_plot,
                                 yield_comparison_plot=yield_comparison_plot,
                                 input_data={
                                     'soil_temp': soil_temp,
                                     'soil_moisture': soil_moisture,
                                     'air_temp': air_temp,
                                     'air_humidity': air_humidity,
                                     'growth_stage': growth_stage,
                                     'purpose': purpose
                                 })
            
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
            
            return render_template('Intelligent_Fertilizer_results.html',
                                 results=results,
                                 confidence_plot=confidence_plot,
                                 yield_comparison_plot=yield_comparison_plot,
                                 input_data={
                                     'soil_temp': soil_temp,
                                     'soil_moisture': soil_moisture,
                                     'air_temp': air_temp,
                                     'air_humidity': air_humidity,
                                     'growth_stage': growth_stage,
                                     'purpose': purpose
                                 },
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
                        predicted_class_name = f"Class_{predicted_class_idx}"
                    
                    # Get recommended action
                    if image_meta and "action_map" in image_meta:
                        raw_action_map = image_meta.get("action_map", {})
                        action_map = {k.lower(): v for k, v in raw_action_map.items()}
                        recommended_action = action_map.get(
                            predicted_class_name.lower(),
                            "Monitor the crop regularly and consult agricultural expert."
                        )
                    else:
                        recommended_action = "Monitor the crop regularly and consult agricultural expert."
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
                
                # Generate confidence plot
                confidence_plot = generate_confidence_plot(confidence_score)
                
                # Read image for display
                with open(filepath, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                
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
            light = float(request.form.get('light', 52300))
            dayofyear = int(request.form.get('dayofyear', datetime.now().timetuple().tm_yday))
            
            # Make prediction
            features = [temperature, humidity, pressure, light, dayofyear]
            result = predict_from_features(features)
            
            # Generate confidence plot
            confidence_plot = generate_confidence_plot(result['probability'])
            
            return render_template('pest_results.html',
                                 result=result,
                                 confidence_plot=confidence_plot,
                                 prediction_type='Sensor-Based',
                                 input_data={
                                     'temperature': temperature,
                                     'humidity': humidity,
                                     'pressure': pressure,
                                     'light': light,
                                     'dayofyear': dayofyear
                                 })
            
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
            light = float(request.form.get('light', 52300))
            dayofyear = int(request.form.get('dayofyear', datetime.now().timetuple().tm_yday))
            
            # Make prediction
            features = [temperature, humidity, pressure, light, dayofyear]
            result = predict_from_features(features)
            
            # Generate confidence plot
            confidence_plot = generate_confidence_plot(result['probability'])
            
            return render_template('pest_results.html',
                                 result=result,
                                 confidence_plot=confidence_plot,
                                 prediction_type='Sensor-Based',
                                 input_data={
                                     'temperature': temperature,
                                     'humidity': humidity,
                                     'pressure': pressure,
                                     'light': light,
                                     'dayofyear': dayofyear
                                 })
            
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
        
        # Parse features
        if 'temperature' in data and 'humidity' in data and 'pressure' in data and 'light' in data:
            features = [
                float(data.get('temperature', 31.2)),
                float(data.get('humidity', 86.5)),
                float(data.get('pressure', 1006.3)),
                float(data.get('light', 52300)),
                int(data.get('dayofyear', datetime.now().timetuple().tm_yday))
            ]
        else:
            return jsonify({'success': False, 'error': 'Invalid input format'}), 400
        
        result = predict_from_features(features)
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ============================================================================
# PRICES DEMAND ROUTES
# ============================================================================

@app.route('/Prices_Demand_predict', methods=['GET', 'POST'])
@login_required
def Prices_Demand_predict():
    if request.method == 'POST':
        try:
            start_date = request.form.get('start_date')
            n_days = int(request.form.get('n_days', 7))

            if not start_date:
                start_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

            start_date = pd.to_datetime(start_date)
            end_date = start_date + timedelta(days=n_days - 1)

            # Use available models or fallback
            if prices_models and prices_models.get('lstm'):
                lstm_features = prices_models.get('lstm_features', ['Paddy_Price_LKR_per_kg'])
                lstm_model = prices_models.get('lstm')
                window_size = prices_models.get('window_size', 21)
            else:
                lstm_features = ['Paddy_Price_LKR_per_kg', 'Demand_Tons', 'Nitrogen_N', 'Phosphorus_P', 'Potassium_K']
                lstm_model = None
                window_size = 21

            # Make price predictions
            price_predictions, prediction_dates = predict_price_with_trend(
                prices_df_mm if prices_df_mm is not None else pd.DataFrame(),
                lstm_model,
                lstm_features,
                start_date,
                n_steps=n_days,
                window_size=window_size
            )

            # Make demand predictions
            if prices_models and prices_models.get('xgb'):
                xgb_model = prices_models.get('xgb')
                xgb_features = prices_models.get('xgb_features', ['Price'])
            else:
                xgb_model = None
                xgb_features = ['Price']

            demand_predictions = predict_demand_with_trend(
                prices_df_std if prices_df_std is not None else pd.DataFrame(),
                xgb_model,
                xgb_features,
                price_predictions,
                prediction_dates
            )

            # Convert prediction_dates to list of strings for template
            date_strings = [d.strftime('%Y-%m-%d') for d in prediction_dates]
            
            # Generate plots
            price_plot = generate_price_plot(prices_df_raw, prediction_dates, price_predictions)
            demand_plot = generate_demand_plot(prices_df_raw, prediction_dates, demand_predictions)
            combined_plot = generate_combined_plot(prices_df_raw, prediction_dates, price_predictions, demand_predictions)

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
    return render_template(
        'Prices_Demand_predict.html',
        today=datetime.now().strftime('%Y-%m-%d'),
        tomorrow=tomorrow
    )

@app.route('/Prices_Demand_sensor_predict', methods=['GET', 'POST'])
@login_required
def Prices_Demand_sensor_predict():
    latest_sensor = sensor_monitor.get_latest() if sensor_monitor else None

    if request.method == 'POST':
        try:
            start_date = request.form.get('start_date')
            n_days = int(request.form.get('n_days', 7))

            nitrogen = float(request.form.get('nitrogen', latest_sensor["values"]["nitrogen"] if latest_sensor else 520))
            phosphorus = float(request.form.get('phosphorus', latest_sensor["values"]["phosphorus"] if latest_sensor else 10))
            potassium = float(request.form.get('potassium', latest_sensor["values"]["potassium"] if latest_sensor else 10))

            if not start_date:
                start_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

            start_date = pd.to_datetime(start_date)
            end_date = start_date + timedelta(days=n_days - 1)

            # Update sensor values in dataframe
            if prices_df_mm is not None and len(prices_df_mm) > 0:
                df_mm_sensor = prices_df_mm.copy()
                df_mm_sensor.loc[df_mm_sensor.index[-1], "Nitrogen_N"] = nitrogen
                df_mm_sensor.loc[df_mm_sensor.index[-1], "Phosphorus_P"] = phosphorus
                df_mm_sensor.loc[df_mm_sensor.index[-1], "Potassium_K"] = potassium
            else:
                # Create a simple dataframe with the sensor values
                df_mm_sensor = pd.DataFrame({
                    'Date': [datetime.now()],
                    'Paddy_Price_LKR_per_kg': [120],
                    'Nitrogen_N': [nitrogen],
                    'Phosphorus_P': [phosphorus],
                    'Potassium_K': [potassium],
                    'DayOfYear': [datetime.now().timetuple().tm_yday],
                    'Month': [datetime.now().month]
                })

            # Make price predictions
            if prices_models and prices_models.get('lstm'):
                lstm_features = prices_models.get('lstm_features', ['Paddy_Price_LKR_per_kg'])
                lstm_model = prices_models.get('lstm')
                window_size = prices_models.get('window_size', 21)
            else:
                lstm_features = ['Paddy_Price_LKR_per_kg', 'Demand_Tons', 'Nitrogen_N', 'Phosphorus_P', 'Potassium_K']
                lstm_model = None
                window_size = 21

            price_predictions, prediction_dates = predict_price_with_trend(
                df_mm_sensor,
                lstm_model,
                lstm_features,
                start_date,
                n_steps=n_days,
                window_size=window_size
            )

            # Make demand predictions
            if prices_models and prices_models.get('xgb'):
                xgb_model = prices_models.get('xgb')
                xgb_features = prices_models.get('xgb_features', ['Price'])
            else:
                xgb_model = None
                xgb_features = ['Price']

            demand_predictions = predict_demand_with_trend(
                prices_df_std if prices_df_std is not None else pd.DataFrame(),
                xgb_model,
                xgb_features,
                price_predictions,
                prediction_dates
            )

            # Convert prediction_dates to list of strings for template
            date_strings = [d.strftime('%Y-%m-%d') for d in prediction_dates]
            
            # Generate plots
            price_plot = generate_price_plot(prices_df_raw, prediction_dates, price_predictions)
            demand_plot = generate_demand_plot(prices_df_raw, prediction_dates, demand_predictions)
            combined_plot = generate_combined_plot(prices_df_raw, prediction_dates, price_predictions, demand_predictions)

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
    return render_template(
        'Prices_Demand_sensor_predict.html',
        today=datetime.now().strftime('%Y-%m-%d'),
        tomorrow=tomorrow,
        latest_sensor=latest_sensor
    )

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
        n_days = int(data.get('n_days', 7))

        # Make price predictions
        if prices_models and prices_models.get('lstm'):
            lstm_features = prices_models.get('lstm_features', ['Paddy_Price_LKR_per_kg'])
            lstm_model = prices_models.get('lstm')
            window_size = prices_models.get('window_size', 21)
        else:
            lstm_features = ['Paddy_Price_LKR_per_kg']
            lstm_model = None
            window_size = 21

        price_predictions, prediction_dates = predict_price_with_trend(
            prices_df_mm if prices_df_mm is not None else pd.DataFrame(),
            lstm_model,
            lstm_features,
            start_date,
            n_steps=n_days,
            window_size=window_size
        )

        # Make demand predictions
        if prices_models and prices_models.get('xgb'):
            xgb_model = prices_models.get('xgb')
            xgb_features = prices_models.get('xgb_features', ['Price'])
        else:
            xgb_model = None
            xgb_features = ['Price']

        demand_predictions = predict_demand_with_trend(
            prices_df_std if prices_df_std is not None else pd.DataFrame(),
            xgb_model,
            xgb_features,
            price_predictions,
            prediction_dates
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

@app.route('/api/reconnect-sensor', methods=['POST'])
@login_required
def api_reconnect_sensor():
    try:
        if sensor_monitor:
            sensor_monitor.stop()
            time.sleep(1)
            sensor_monitor.start()
            return jsonify({
                "success": True,
                "sensor": sensor_monitor.get_latest()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Sensor monitor not initialized"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/debug-sensor')
@login_required
def debug_sensor():
    if sensor_monitor:
        return jsonify(sensor_monitor.get_latest())
    else:
        return jsonify({"error": "Sensor monitor not initialized"}), 500

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