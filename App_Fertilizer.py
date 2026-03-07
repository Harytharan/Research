from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import sys
import joblib
import numpy as np
import pandas as pd
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTELLIGENT_FERTILIZER_DIR = os.path.join(BASE_DIR, 'Intelligent_Fertilizer')

#  path
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, INTELLIGENT_FERTILIZER_DIR)


try:
    current_dir = os.getcwd()
    os.chdir(INTELLIGENT_FERTILIZER_DIR)
    print(f" Changed to directory: {os.getcwd()}")

    # Change back to original directory
    os.chdir(current_dir)
    print(f" Changed back to: {os.getcwd()}")
    
    print(" Intelligent_Fertilizer modules imported successfully")
except ImportError as e:
    print(f" Warning: Could not import modules: {e}")
except Exception as e:
    print(f" Error: {e}")

app = Flask(__name__,
            template_folder=os.path.join(INTELLIGENT_FERTILIZER_DIR, 'templates'),
            static_folder=os.path.join(INTELLIGENT_FERTILIZER_DIR, 'static'))

app.secret_key = 'fertilizer_recommendation_secret_key_2026'
app.config['UPLOAD_FOLDER'] = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(INTELLIGENT_FERTILIZER_DIR, 'static', 'images'), exist_ok=True)
os.makedirs(os.path.join(INTELLIGENT_FERTILIZER_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(INTELLIGENT_FERTILIZER_DIR, 'fertilizer_models'), exist_ok=True)
os.makedirs(os.path.join(INTELLIGENT_FERTILIZER_DIR, 'yield_models'), exist_ok=True)

# Global variables
clf = None
reg = None
scaler = None
encoders = None
feature_names = None
df_master = None

def load_models():
    global clf, reg, scaler, encoders, feature_names, df_master
    
    try:
        print(f"\n🔍 Checking model files in: {INTELLIGENT_FERTILIZER_DIR}")
        
        # Load models from Intelligent_Fertilizer directory
        clf_path = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'fertilizer_models', 'fertilizer_model.pkl')
        reg_path = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'yield_models', 'yield_model.pkl')
        scaler_path = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'yield_models', 'scaler.pkl')
        encoders_path = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'fertilizer_models', 'label_encoders.pkl')
        features_path = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'fertilizer_models', 'feature_names.pkl')
        
        print(f"   Classifier path: {clf_path}")
        print(f"   Classifier exists: {os.path.exists(clf_path)}")
        print(f"   Regressor path: {reg_path}")
        print(f"   Regressor exists: {os.path.exists(reg_path)}")
        
        if os.path.exists(clf_path):
            clf = joblib.load(clf_path)
            print(" Fertilizer classifier loaded")
        else:
            print(f" Classifier not found")
            return False
        
        if os.path.exists(reg_path):
            reg = joblib.load(reg_path)
            print(" Yield regressor loaded")
        else:
            print(f" Regressor not found")
            return False
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(" Scaler loaded")
        
        if os.path.exists(encoders_path):
            encoders = joblib.load(encoders_path)
            print(" Label encoders loaded")
            

            if "Paddy_Growth_Stage" in encoders:
                print(f"   Growth stages: {list(encoders['Paddy_Growth_Stage'].classes_)}")
            if "Purpose" in encoders:
                print(f"   Purposes: {list(encoders['Purpose'].classes_)}")
            if "Recommended_Fertilizer" in encoders:
                print(f"   Fertilizers: {list(encoders['Recommended_Fertilizer'].classes_)}")
        
        if os.path.exists(features_path):
            feature_names = joblib.load(features_path)
            print(f" Feature names loaded: {len(feature_names)} features")
            print(f"   Features: {feature_names}")
        
        # Load master dataset for quantity and sustainability notes
        dataset_path = os.path.join(INTELLIGENT_FERTILIZER_DIR, 'Dataset.csv')
        if os.path.exists(dataset_path):
            df_master = pd.read_csv(dataset_path)
            print(f" Master dataset loaded: {len(df_master)} records")
        else:
            print(f" Dataset not found at: {dataset_path}")
        
        return True
        
    except Exception as e:
        print(f" Error loading models: {e}")
        return False

def predict_top3(soil_temp, soil_moisture, air_temp, air_humidity, growth_stage, purpose):

    
    if clf is None or reg is None or scaler is None or encoders is None:
        raise Exception("Models not loaded. Please train the models first.")
    
    # Encode categorical inputs
    try:
        if growth_stage not in encoders["Paddy_Growth_Stage"].classes_:
            valid_stages = list(encoders["Paddy_Growth_Stage"].classes_)
            raise Exception(f"Invalid growth stage. Must be one of: {valid_stages}")
        
        if purpose not in encoders["Purpose"].classes_:
            valid_purposes = list(encoders["Purpose"].classes_)
            raise Exception(f"Invalid purpose. Must be one of: {valid_purposes}")
        
        growth_stage_encoded = encoders["Paddy_Growth_Stage"].transform([growth_stage])[0]
        purpose_encoded = encoders["Purpose"].transform([purpose])[0]
        
    except Exception as e:
        raise Exception(f"Encoding error: {str(e)}")
    

    base = {
        "Soil_Temperature (°C)": soil_temp,
        "Soil_Moisture (%)": soil_moisture,
        "Air_Temperature (°C)": air_temp,
        "Air_Humidity (%)": air_humidity,
        "Paddy_Growth_Stage": growth_stage_encoded,
        "Purpose": purpose_encoded,
        "Quantity_kg_per_acre": 25,  # temporary
        "Recommended_Fertilizer": 0   # dummy for probability
    }
    
    # Create DataFrame for probability prediction
    X_base = pd.DataFrame([base])[feature_names]
    X_scaled = scaler.transform(X_base)
    
    # Get probabilities
    probs = clf.predict_proba(X_scaled)[0]
    top3_indices = probs.argsort()[-3:][::-1]
    
    results = []
    
    for rank, fert_idx in enumerate(top3_indices, start=1):
        fert_name = encoders["Recommended_Fertilizer"].inverse_transform([fert_idx])[0]
        
        # Get quantity and sustainability note from master dataset
        if df_master is not None:
            match = df_master[df_master["Recommended_Fertilizer"] == fert_name]
            if not match.empty:
                quantity = float(match.iloc[0]["Quantity_kg_per_acre"])
                sustainability_note = str(match.iloc[0]["Sustainability_Note"])
            else:
                quantity = 25.0
                sustainability_note = "Follow standard application practices."
        else:
            quantity = 25.0
            sustainability_note = "Follow standard application practices."
        

        temp = base.copy()
        temp["Quantity_kg_per_acre"] = quantity
        temp["Recommended_Fertilizer"] = fert_idx
        
        X = pd.DataFrame([temp])[feature_names]
        X_scaled = scaler.transform(X)
        
        yield_pred = float(reg.predict(X_scaled)[0])
        cost_pred = yield_pred * 25000  # Estimated cost per ton
        
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

def initialize_app():

    print("\n" + "="*60)
    print(" INTELLIGENT FERTILIZER RECOMMENDATION SYSTEM")
    print("="*60)
    print(f" Base directory: {BASE_DIR}")
    print(f" Intelligent_Fertilizer directory: {INTELLIGENT_FERTILIZER_DIR}")
    print(f" Templates directory: {app.template_folder}")
    print(f" Static directory: {app.static_folder}")
    
    # Check if Intelligent_Fertilizer directory exists
    if os.path.exists(INTELLIGENT_FERTILIZER_DIR):
        print(f"\n Contents of Intelligent_Fertilizer folder:")
        for item in os.listdir(INTELLIGENT_FERTILIZER_DIR):
            print(f"   - {item}")
    else:
        print(f"\n Intelligent_Fertilizer folder not found at: {INTELLIGENT_FERTILIZER_DIR}")
    
    # Load models
    print("\n Loading models...")
    models_loaded = load_models()
    
    print("\n" + "="*60)
    if models_loaded:
        print(" Application initialized successfully!")
    else:
        print(" Models not loaded. Please train the models first.")
        print("   Run: python train.py in the Intelligent_Fertilizer folder")
    print("="*60)

# Initialize on startup
initialize_app()

@app.context_processor
def utility_processor():
    """Add utility functions to template context"""
    return {
        'now': datetime.now(),
        'app_name': 'Intelligent Fertilizer Recommendation',
        'models_loaded': clf is not None and reg is not None
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        try:

            if clf is None or reg is None:
                flash('Models not loaded. Please train the models first.', 'error')
                return redirect(url_for('predict'))
            
            # Get form data
            soil_temp = float(request.form.get('soil_temp'))
            soil_moisture = float(request.form.get('soil_moisture'))
            air_temp = float(request.form.get('air_temp'))
            air_humidity = float(request.form.get('air_humidity'))
            growth_stage = request.form.get('growth_stage')
            purpose = request.form.get('purpose')
            

            results = predict_top3(
                soil_temp, soil_moisture, air_temp, air_humidity,
                growth_stage, purpose
            )
            

            confidence_plot = generate_confidence_plot(results)
            yield_comparison_plot = generate_yield_comparison_plot(results)
            
            return render_template('results.html',
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
            return redirect(url_for('predict'))
    

    growth_stages = []
    purposes = []
    
    if encoders is not None:
        if "Paddy_Growth_Stage" in encoders:
            growth_stages = list(encoders["Paddy_Growth_Stage"].classes_)
        if "Purpose" in encoders:
            purposes = list(encoders["Purpose"].classes_)
    
    return render_template('predict.html',
                         growth_stages=growth_stages,
                         purposes=purposes)

@app.route('/sensor-predict', methods=['GET', 'POST'])
def sensor_predict():

    if request.method == 'POST':
        try:

            if clf is None or reg is None:
                flash('Models not loaded. Please train the models first.', 'error')
                return redirect(url_for('sensor_predict'))
            
            # Get form data
            soil_temp = float(request.form.get('soil_temp'))
            soil_moisture = float(request.form.get('soil_moisture'))
            air_temp = float(request.form.get('air_temp'))
            air_humidity = float(request.form.get('air_humidity'))
            growth_stage = request.form.get('growth_stage')
            purpose = request.form.get('purpose')
            

            results = predict_top3(
                soil_temp, soil_moisture, air_temp, air_humidity,
                growth_stage, purpose
            )
            

            confidence_plot = generate_confidence_plot(results)
            yield_comparison_plot = generate_yield_comparison_plot(results)
            
            return render_template('results.html',
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
            return redirect(url_for('sensor_predict'))
    

    growth_stages = []
    purposes = []
    
    if encoders is not None:
        if "Paddy_Growth_Stage" in encoders:
            growth_stages = list(encoders["Paddy_Growth_Stage"].classes_)
        if "Purpose" in encoders:
            purposes = list(encoders["Purpose"].classes_)
    
    return render_template('sensor_predict.html',
                         growth_stages=growth_stages,
                         purposes=purposes)



@app.route('/api/predict', methods=['POST'])
def api_predict():

    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validate required fields
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

def generate_confidence_plot(results):

    plt.figure(figsize=(8, 5))
    
    fertilizers = [r['fertilizer'] for r in results]
    confidences = [r['confidence'] for r in results]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars = plt.bar(fertilizers, confidences, color=colors, alpha=0.8)
    plt.ylabel('Confidence (%)', fontsize=12)
    plt.title('Fertilizer Recommendation Confidence', fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 100)
    
    # Add value labels
    for bar, conf in zip(bars, confidences):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{conf}%', ha='center', va='bottom', fontsize=11)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Convert to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def generate_yield_comparison_plot(results):

    plt.figure(figsize=(8, 5))
    
    fertilizers = [r['fertilizer'] for r in results]
    yields = [r['yield'] for r in results]
    colors = ['#3498db', '#9b59b6', '#1abc9c']
    
    bars = plt.bar(fertilizers, yields, color=colors, alpha=0.8)
    plt.ylabel('Predicted Yield (ton/ha)', fontsize=12)
    plt.title('Expected Yield by Fertilizer', fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
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
    app.run(debug=True, host='0.0.0.0', port=5002) 