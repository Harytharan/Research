# live_predict_once_clean.py
import serial
import time
import traceback
import sys
import joblib
import pandas as pd

# ---------------- Predict Function ----------------
def predict_total_cost(input_data):
    """
    Predict total paddy farming cost for given input data (dict).
    """
    model = joblib.load("trained_paddy_cost_model.pkl")
    encoders = joblib.load("label_encoders.pkl")

    df_input = pd.DataFrame([input_data])

    # Encode categorical columns using saved encoders
    for col, le in encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col])

    # Predict cost
    predicted_cost = model.predict(df_input)[0]
    return round(predicted_cost, 2)


# ---------------- Read Sensor Once ----------------
def read_sensor_once(port="COM3", baudrate=115200, timeout=5):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Connected to {port} at {baudrate} baud")
    except Exception:
        print("ERROR: Cannot open serial port!")
        traceback.print_exc()
        sys.exit(1)

    sensor_data = {}
    start_time = time.time()

    while time.time() - start_time < timeout:  # read for max `timeout` seconds
        try:
            raw = ser.readline()
            if not raw:
                continue

            try:
                text = raw.decode('utf-8').strip()
            except UnicodeDecodeError:
                text = raw.decode(errors="ignore").strip()

            # ---------------- Parse Humidity & Temp safely ----------------
            if "Humidity" in text and "Temperature" in text:
                try:
                    # Only parse if numbers are present
                    if any(c.isdigit() for c in text):
                        # Extract Humidity
                        hum_str = text.split("Humidity")[1].split(":")[1].split("%")[0].strip()
                        # Extract Temperature Celsius
                        temp_str = text.split("Temperature")[1].split(":")[1].split("°")[0].strip()

                        sensor_data["Humidity_%"] = float(hum_str)
                        sensor_data["Temperature_C"] = float(temp_str)

                except Exception:
                    # Ignore bad lines silently
                    pass

            # Parse other key:value lines
            elif ":" in text:
                key, value = text.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    sensor_data[key] = float(value)
                except:
                    sensor_data[key] = value

            # Stop if all required values collected
            if all(k in sensor_data for k in ["Latitude", "Longitude", "Temperature_C", "Humidity_%"]):
                break

        except Exception:
            print("Serial Read Error:")
            traceback.print_exc()

    ser.close()
    return sensor_data


# ---------------- Main ----------------
if __name__ == "__main__":
    data = read_sensor_once(port="COM3", baudrate=115200, timeout=10)
    print("Sensor Data Collected:", data)

    if all(k in data for k in ["Latitude", "Longitude", "Temperature_C", "Humidity_%"]):
        sample_input = {
            "Latitude": data["Latitude"],
            "Longitude": data["Longitude"],
            "Year": 2025,
            "Season": "Maha",
            "Soil_Type": "Clay",
            "Rainfall_mm": 2200,
            "Temperature_C": data["Temperature_C"],
            "Humidity_%": data["Humidity_%"],
            "Area_acres": 2.5,
            "Seed_Type": "BG352",
            "Seed_Cost (LKR)": 8000,
            "Fertilizer_Cost (LKR)": 15000,
            "Pesticide_Cost (LKR)": 4000,
            "Labor_Cost (LKR)": 25000,
            "Water_Cost (LKR)": 5000,
            "Machinery_Cost (LKR)": 10000,
            "Other_Costs (LKR)": 2000
        }
        try:
            cost = predict_total_cost(sample_input)
            print(f"\nPredicted Total Cost: LKR {cost}\n")
        except Exception:
            print("Prediction Error:")
            traceback.print_exc()
    else:
        print("ERROR: Could not get all required sensor values.")
