# utils/data_utils.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_historical_data(df_raw):
    """Get summary statistics of historical data"""
    if df_raw is None or len(df_raw) == 0:
        return {}
    
    try:
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        
        historical_data = {
            'date_range': {
                'start': df_raw['Date'].min().strftime('%Y-%m-%d'),
                'end': df_raw['Date'].max().strftime('%Y-%m-%d'),
                'total_days': len(df_raw)
            },
            'price_stats': {
                'mean': float(df_raw['Paddy_Price_LKR_per_kg'].mean()),
                'min': float(df_raw['Paddy_Price_LKR_per_kg'].min()),
                'max': float(df_raw['Paddy_Price_LKR_per_kg'].max()),
                'std': float(df_raw['Paddy_Price_LKR_per_kg'].std())
            },
            'demand_stats': {
                'mean': float(df_raw['Demand_Tons'].mean()),
                'min': float(df_raw['Demand_Tons'].min()),
                'max': float(df_raw['Demand_Tons'].max()),
                'std': float(df_raw['Demand_Tons'].std())
            },
            'regions': df_raw['Region'].unique().tolist() if 'Region' in df_raw.columns else [],
            'recent_trend': calculate_recent_trend(df_raw)
        }
        
        return historical_data
    except Exception as e:
        print(f"Error getting historical data: {e}")
        return {}

def calculate_recent_trend(df_raw, days=30):
    """Calculate recent trend direction"""
    try:
        recent_data = df_raw.tail(days)
        
        price_trend = 'Stable'
        if len(recent_data) > 1:
            price_change = (recent_data['Paddy_Price_LKR_per_kg'].iloc[-1] - 
                          recent_data['Paddy_Price_LKR_per_kg'].iloc[0])
            if price_change > 1:
                price_trend = 'Increasing'
            elif price_change < -1:
                price_trend = 'Decreasing'
        
        demand_trend = 'Stable'
        if len(recent_data) > 1:
            demand_change = (recent_data['Demand_Tons'].iloc[-1] - 
                           recent_data['Demand_Tons'].iloc[0])
            if demand_change > 5:
                demand_trend = 'Increasing'
            elif demand_change < -5:
                demand_trend = 'Decreasing'
        
        return {
            'price': price_trend,
            'demand': demand_trend
        }
    except:
        return {'price': 'Unknown', 'demand': 'Unknown'}

def save_prediction_results(predictions, filename=None):
    """Save prediction results to CSV"""
    if filename is None:
        filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    df = pd.DataFrame(predictions)
    df.to_csv(filename, index=False)
    return filename

def validate_sensor_values(n, p, k):
    """Validate sensor values and return warnings"""
    warnings = []
    
    if n < 400 or n > 650:
        warnings.append(f"Nitrogen value {n} is outside typical range (400-650)")
    if p < 3 or p > 20:
        warnings.append(f"Phosphorus value {p} is outside typical range (3-20)")
    if k < 3 or k > 20:
        warnings.append(f"Potassium value {k} is outside typical range (3-20)")
    
    return warnings