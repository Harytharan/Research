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
from datetime import datetime, timedelta
import io
import base64
import warnings
warnings.filterwarnings('ignore')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FARMING_COST_DIR = os.path.join(BASE_DIR, 'Farming_Cost')


sys.path.insert(0, FARMING_COST_DIR)
try:
    from load import load_dataset
    from pre_process import preprocess_data
    from featureeng import create_features
    from predict import predict_total_cost
    print("Successfully imported all modules from Farming_Cost")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Looking in: {FARMING_COST_DIR}")
    print(f"Files in directory: {os.listdir(FARMING_COST_DIR)}")
    sys.exit(1)

app = Flask(__name__,
            template_folder=os.path.join(FARMING_COST_DIR, 'templates'),
            static_folder=os.path.join(FARMING_COST_DIR, 'static'))

app.secret_key = 'paddy_farming_cost_secret_key_2024'
app.config['UPLOAD_FOLDER'] = os.path.join(FARMING_COST_DIR, 'static', 'uploads')

#  directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(FARMING_COST_DIR, 'static', 'images'), exist_ok=True)
os.makedirs(os.path.join(FARMING_COST_DIR, 'models'), exist_ok=True)

# Global variables
model = None
encoders = None
historical_data = None
df_raw = None

def verify_model_files():
    model_path = os.path.join(FARMING_COST_DIR, 'models', 'trained_paddy_cost_model.pkl')
    encoders_path = os.path.join(FARMING_COST_DIR, 'models', 'label_encoders.pkl')
    
    print(f"\n Checking model files:")
    print(f"   Model path: {model_path}")
    print(f"   Model exists: {os.path.exists(model_path)}")
    print(f"   Encoders path: {encoders_path}")
    print(f"   Encoders exist: {os.path.exists(encoders_path)}")
    
    return os.path.exists(model_path) and os.path.exists(encoders_path)

def load_historical_data():
    global df_raw, historical_data
    
    try:
        # Save current directory
        current_dir = os.getcwd()
        
        # Change to Farming_Cost directory temporarily
        os.chdir(FARMING_COST_DIR)
        print(f" Changed to directory: {os.getcwd()}")
        
        # Check if dataset exists
        dataset_path = os.path.join(FARMING_COST_DIR, 'paddy_farming_cost_dataset.csv')
        if not os.path.exists(dataset_path):
            print(f" Dataset not found at: {dataset_path}")
            os.chdir(current_dir)
            return False
        
        # Load dataset
        df_raw = load_dataset()
        
        # Change back to original directory
        os.chdir(current_dir)
        print(f" Changed back to: {os.getcwd()}")
        
        if df_raw is not None:
            historical_data = get_historical_data(df_raw)
            print(f" Loaded {len(df_raw)} historical records")
            return True
        else:
            print("⚠ Could not load historical data")
            return False
    except Exception as e:
        print(f" Error loading data: {e}")
        return False

def get_historical_data(df):
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
        
        return historical_data
    except Exception as e:
        print(f" Error getting historical data: {e}")
        return {}

def initialize_app():
    """Initialize application on startup"""
    global model, encoders, df_raw, historical_data
    
    print("\n" + "="*60)
    print("🌾 PADDY FARMING COST ESTIMATION SYSTEM")
    print("="*60)
    print(f" Base directory: {BASE_DIR}")
    print(f" Farming_Cost directory: {FARMING_COST_DIR}")
    print(f" Templates directory: {app.template_folder}")
    print(f" Static directory: {app.static_folder}")
    
    # Verify model files exist
    models_exist = verify_model_files()
    
    if not models_exist:
        print("\n Model files not found!")
        print("   Please train the model first by running:")
        print(f"   cd {FARMING_COST_DIR}")
        print("   python train.py")
        print("   cd ..")
    else:
        print("\n Model files verified")
    
    # Load historical data
    print("\n Loading historical data...")
    data_loaded = load_historical_data()
    
    print("\n" + "="*60)
    if models_exist and data_loaded:
        print(" Application initialized successfully!")
    else:
        print(" Application initialized with warnings!")
        if not models_exist:
            print("   - Model files missing (run train.py)")
        if not data_loaded:
            print("   - Historical data not loaded")
    print("="*60)

# Initialize on startup
initialize_app()

@app.context_processor
def utility_processor():
    return {
        'now': datetime.now(),
        'app_name': 'Paddy Farming Cost Estimator',
        'model_loaded': verify_model_files()
    }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', 
                         historical_data=historical_data,
                         model_loaded=verify_model_files())

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            if not verify_model_files():
                flash('Model not trained. Please train the model first using: python train.py', 'error')
                return redirect(url_for('predict'))
            
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
            
            return render_template('results.html',
                                 results=results,
                                 cost_breakdown_plot=cost_breakdown_plot,
                                 comparison_plot=comparison_plot)
            
        except FileNotFoundError as e:
            flash(f'Model files not found: {str(e)}. Please train the model first.', 'error')
            return redirect(url_for('predict'))
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
            return redirect(url_for('predict'))
    
    # GET request - show form
    return render_template('predict.html',
                         current_year=datetime.now().year,
                         soil_types=historical_data.get('soil_types', ['Clay', 'Sandy Loam', 'Loam', 'Silt']),
                         seed_types=historical_data.get('seed_types', ['BG300', 'BG352', 'BG358', 'At306', 'At402']),
                         seasons=historical_data.get('seasons', ['Maha', 'Yala']))

@app.route('/sensor-predict', methods=['GET', 'POST'])
def sensor_predict():
    if request.method == 'POST':
        try:
            # Verify model exists before prediction
            if not verify_model_files():
                flash('Model not trained. Please train the model first using: python train.py', 'error')
                return redirect(url_for('sensor_predict'))
            
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
            
            return render_template('results.html',
                                 results=results,
                                 cost_breakdown_plot=cost_breakdown_plot,
                                 comparison_plot=comparison_plot)
            
        except FileNotFoundError as e:
            flash(f'Model files not found: {str(e)}. Please train the model first.', 'error')
            return redirect(url_for('sensor_predict'))
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            import traceback
            traceback.print_exc()
            return redirect(url_for('sensor_predict'))
    
    # GET request - show form
    return render_template('sensor_predict.html',
                         current_year=datetime.now().year,
                         soil_types=historical_data.get('soil_types', ['Clay', 'Sandy Loam', 'Loam', 'Silt']),
                         seed_types=historical_data.get('seed_types', ['BG300', 'BG352', 'BG358', 'At306', 'At402']),
                         seasons=historical_data.get('seasons', ['Maha', 'Yala']))



@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        if not verify_model_files():
            return jsonify({'success': False, 'error': 'Model not trained'}), 500
        
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

def generate_cost_breakdown_plot(cost_breakdown, predicted_cost):
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
    plt.figure(figsize=(8, 6))
    
    categories = ['Input Total', 'Predicted Cost']
    values = [total_input, predicted_cost]
    colors = ['#3498db', '#2ecc71']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.8)
    plt.ylabel('Cost (LKR)', fontsize=12)
    plt.title('Input vs Predicted Total Cost', fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
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

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("\n Starting Flask application...")
    print(f" Access the application at: http://localhost:5000")
    print(" Press CTRL+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5001) 
    