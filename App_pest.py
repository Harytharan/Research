from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import sys
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64
import json
import warnings
warnings.filterwarnings('ignore')
from werkzeug.utils import secure_filename
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PEST_OUTBREAK_DIR = os.path.join(BASE_DIR, 'Pest_outbreak')


sys.path.insert(0, BASE_DIR)
sys.path.insert(0, PEST_OUTBREAK_DIR)


try:
    # Save current directory
    current_dir = os.getcwd()
    
    # Change to Pest_outbreak directory
    os.chdir(PEST_OUTBREAK_DIR)
    print(f" Changed to directory: {os.getcwd()}")
    
    # Now import the modules
    import predict_image as prt
    import visualize_prediction as Visual
    
    # Change back to original directory
    os.chdir(current_dir)
    print(f" Changed back to: {os.getcwd()}")
    
    print(" Image modules imported successfully")
except ImportError as e:
    print(f" Warning: Could not import Image modules: {e}")
    prt = None
    Visual = None
except Exception as e:
    print(f" Error: {e}")
    prt = None
    Visual = None

app = Flask(__name__,
            template_folder=os.path.join(PEST_OUTBREAK_DIR, 'templates'),
            static_folder=os.path.join(PEST_OUTBREAK_DIR, 'static'))

app.secret_key = 'paddy_pest_detection_secret_key_2026'
app.config['UPLOAD_FOLDER'] = os.path.join(PEST_OUTBREAK_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(PEST_OUTBREAK_DIR, 'static', 'images'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global variables
tabular_model = None
tabular_meta = None
image_model = None
image_meta = None
image_class_labels = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_tabular_model():

    global tabular_model, tabular_meta
    
    try:
        model_path = os.path.join(PEST_OUTBREAK_DIR, 'models', 'paddy_pest_rf.joblib')
        meta_path = os.path.join(PEST_OUTBREAK_DIR, 'models', 'paddy_meta.joblib')
        
        if os.path.exists(model_path) and os.path.exists(meta_path):
            tabular_model = joblib.load(model_path)
            tabular_meta = joblib.load(meta_path)
            print(" Tabular model loaded successfully")
            return True
        else:
            print(f" Tabular model files not found at:")
            print(f"   - {model_path}")
            print(f"   - {meta_path}")
            return False
    except Exception as e:
        print(f" Error loading tabular model: {e}")
        return False

def load_image_model():
    global image_model, image_meta, image_class_labels
    
    try:
        model_path = os.path.join(PEST_OUTBREAK_DIR, 'best_pest_model.h5')
        meta_path = os.path.join(PEST_OUTBREAK_DIR, 'paddy_meta.joblib')
        class_indices_path = os.path.join(PEST_OUTBREAK_DIR, 'class_indices.json')
        
        print(f"\n Checking image model files:")
        print(f"   Model path: {model_path}")
        print(f"   Model exists: {os.path.exists(model_path)}")
        print(f"   Class indices path: {class_indices_path}")
        print(f"   Class indices exist: {os.path.exists(class_indices_path)}")
        print(f"   Meta path: {meta_path}")
        print(f"   Meta exists: {os.path.exists(meta_path)}")
        
        if os.path.exists(model_path):
            # Load model
            image_model = load_model(model_path)
            image_meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
            
            if prt is not None and hasattr(prt, 'load_class_labels'):
                image_class_labels = prt.load_class_labels()
                print(f" Found {len(image_class_labels)} pest classes: {image_class_labels}")
            else:
                print(" Image label loader unavailable, using metadata fallback")
                if 'class_names' in image_meta:
                    image_class_labels = image_meta['class_names']
                else:
                    image_class_labels = []
            
            print(" Image model loaded successfully")
            return True
        else:
            print(f" Image model files not found")
            return False
    except Exception as e:
        print(f" Error loading image model: {e}")
        return False

def predict_from_features(values):

    if tabular_model is None or tabular_meta is None:
        raise Exception("Tabular model not loaded")
    

    scaler = tabular_meta["scaler"]
    expected_features = scaler.n_features_in_
    
    print(f" Debug: Expected features: {expected_features}, Received: {len(values)}")
    

    if len(values) > expected_features:
        values = values[:expected_features]
        print(f"   Truncated to {expected_features} features")
    elif len(values) < expected_features:
        # Pad with zeros or mean values
        values = values + [0] * (expected_features - len(values))
        print(f"   Padded to {expected_features} features")
    
    arr = np.array(values, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(arr)
    
    pred_idx = tabular_model.predict(X_scaled)[0]
    label = tabular_meta["label_encoder"].inverse_transform([pred_idx])[0]
    
    try:
        prob = float(np.max(tabular_model.predict_proba(X_scaled)[0]))
    except:
        prob = None
    

    action_map = tabular_meta.get("action_map", {})
    action = action_map.get(label, "No recommended action found.")
    
    return {
        "predicted_pest": label,
        "probability": prob,
        "recommended_action": action
    }

def parse_sensor_string(s):
    parts = s.split(",")
    if len(parts) != 5:
        raise ValueError("Need 5 values: temperature,humidity,pressure,light,dayofyear")
    return [float(p.strip()) for p in parts]

def initialize_app():
    """Initialize application on startup"""
    print("\n" + "="*60)
    print(" PADDY PEST DETECTION SYSTEM")
    print("="*60)
    print(f" Base directory: {BASE_DIR}")
    print(f" Pest_outbreak directory: {PEST_OUTBREAK_DIR}")
    print(f" Templates directory: {app.template_folder}")
    print(f" Static directory: {app.static_folder}")
    
    # Check if directories exist
    print(f"\n Checking directories:")
    print(f"   Base exists: {os.path.exists(BASE_DIR)}")
    print(f"   Pest_outbreak exists: {os.path.exists(PEST_OUTBREAK_DIR)}")
    
    if os.path.exists(PEST_OUTBREAK_DIR):
        print(f"   Files in Pest_outbreak: {os.listdir(PEST_OUTBREAK_DIR)}")
    
    # Load models
    print("\n Loading models...")
    tabular_loaded = load_tabular_model()
    image_loaded = load_image_model()
    
    print("\n" + "="*60)
    if tabular_loaded and image_loaded:
        print(" All models loaded successfully!")
    else:
        print(" Some models could not be loaded:")
        if not tabular_loaded:
            print("   - Tabular model (run train_tabular.py)")
        if not image_loaded:
            print("   - Image model (run train_image.py)")
    print("="*60)

# Initialize on startup
initialize_app()

@app.context_processor
def utility_processor():

    return {
        'now': datetime.now(),
        'app_name': 'Paddy Pest Detection System',
        'tabular_loaded': tabular_model is not None,
        'image_loaded': image_model is not None
    }

@app.route('/')
def index():

    return render_template('index.html',
                         tabular_loaded=tabular_model is not None,
                         image_loaded=image_model is not None)

@app.route('/sensor-predict', methods=['GET', 'POST'])
def sensor_predict():
    if request.method == 'POST':
        try:
            # Check if model is loaded
            if tabular_model is None:
                flash('Tabular model not loaded. Please train the model first.', 'error')
                return redirect(url_for('sensor_predict'))
            
            # Get form data
            temperature = float(request.form.get('temperature', 31.2))
            humidity = float(request.form.get('humidity', 86.5))
            pressure = float(request.form.get('pressure', 1006.3))
            light = float(request.form.get('light', 52300))
            dayofyear = int(request.form.get('dayofyear', datetime.now().timetuple().tm_yday))
            
            # Make prediction
            features = [temperature, humidity, pressure, light, dayofyear]
            result = predict_from_features(features)
            
            # Generate plots
            confidence_plot = generate_confidence_plot(result['probability'])
            
            return render_template('results.html',
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
            return redirect(url_for('sensor_predict'))
    
    # GET request - show form
    return render_template('sensor_predict.html',
                         current_day=datetime.now().timetuple().tm_yday)

@app.route('/image-predict', methods=['GET', 'POST'])
def image_predict():
    if request.method == 'POST':
        try:
            if image_model is None or prt is None:
                flash('Image model not loaded. Please train the model first.', 'error')
                return redirect(url_for('image_predict'))

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
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                predicted_class_idx, confidence_score = prt.predict_image(filepath)

                if image_class_labels and predicted_class_idx < len(image_class_labels):
                    predicted_class_name = image_class_labels[predicted_class_idx]
                else:
                    predicted_class_name = f"Unknown class {predicted_class_idx}"

                if hasattr(prt, 'NON_PEST_LABELS') and predicted_class_name.lower() in prt.NON_PEST_LABELS:
                    predicted_class_name = 'Non-pest item'
                
                # Get recommended action
                raw_action_map = image_meta.get("action_map", {}) if image_meta else {}
                action_map = {k.lower(): v for k, v in raw_action_map.items()}
                recommended_action = action_map.get(
                    predicted_class_name.lower(),
                    "No pest detected in the image." if predicted_class_name == 'Non-pest item' else "No recommended action found."
                )
                
                result = {
                    'predicted_pest': predicted_class_name,
                    'probability': float(confidence_score),
                    'recommended_action': recommended_action
                }
                
                # Generate plots
                confidence_plot = generate_confidence_plot(confidence_score)
                

                with open(filepath, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                
                return render_template('results.html',
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
            import traceback
            traceback.print_exc()
            return redirect(url_for('image_predict'))
    

    return render_template('image_predict.html')

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    try:
        if tabular_model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file, header=None)
        results = []
        
        for i, row in df.iterrows():
            feats = row.values[:5]
            try:
                res = predict_from_features(feats)
                res["row_index"] = int(i)
                results.append(res)
            except Exception as e:
                results.append({
                    "row_index": int(i),
                    "error": str(e)
                })
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400



@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        if tabular_model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        
        # Parse features
        if 'sample' in data:
            features = parse_sensor_string(data['sample'])
        elif all(k in data for k in ['temperature', 'humidity', 'pressure', 'light']):
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

def generate_confidence_plot(confidence):
    plt.figure(figsize=(6, 3))
    
    # Create a horizontal bar for confidence
    categories = ['Confidence']
    values = [confidence * 100]
    colors = ['#2ecc71' if confidence > 0.7 else '#f39c12' if confidence > 0.4 else '#e74c3c']
    
    bars = plt.barh(categories, values, color=colors, alpha=0.8, height=0.5)
    plt.xlim(0, 100)
    plt.xlabel('Confidence (%)', fontsize=11)
    plt.title('Prediction Confidence', fontsize=13, fontweight='bold', pad=15)
    
    # Add value label
    for bar, val in zip(bars, values):
        width = bar.get_width()
        plt.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Convert to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

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

if __name__ == '__main__':
    print("\n Starting Flask application...")
    print(f" Access the application at: http://localhost:5000")
    print(" Press CTRL+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5003)  # Changed from 5000 to 5003