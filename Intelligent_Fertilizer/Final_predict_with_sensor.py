import serial
import traceback
import sys
import os
import joblib
import pandas as pd


# ================= LOAD MODELS =================

MODEL_DIR = ""

clf = joblib.load(os.path.join(MODEL_DIR, "fertilizer_models/fertilizer_model.pkl"))
reg = joblib.load(os.path.join(MODEL_DIR, "yield_models/yield_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "yield_models/scaler.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "fertilizer_models/label_encoders.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "fertilizer_models/feature_names.pkl"))

# Load original dataset (for quantity + sustainability note)
MASTER_DATA = "Dataset.csv"
df_master = pd.read_csv(MASTER_DATA)


# ================= PREDICTION FUNCTION =================

def predict_top3(sensor_data, growth_stage, purpose):

    base = {
        "Soil_Temperature (°C)": sensor_data["soil_temp"],
        "Soil_Moisture (%)": sensor_data["soil_moisture"],
        "Air_Temperature (°C)": sensor_data["air_temp"],
        "Air_Humidity (%)": sensor_data["air_humidity"],
        "Paddy_Growth_Stage": encoders["Paddy_Growth_Stage"].transform([growth_stage])[0],
        "Purpose": encoders["Purpose"].transform([purpose])[0],
        "Quantity_kg_per_acre": 25  # temporary
    }

    # Probability prediction
    temp_df = pd.DataFrame([{**base, "Recommended_Fertilizer": 0}])[feature_names]
    probs = clf.predict_proba(scaler.transform(temp_df))[0]
    top3 = probs.argsort()[-3:][::-1]

    results = []

    for rank, fert_id in enumerate(top3, start=1):
        fert_name = encoders["Recommended_Fertilizer"].inverse_transform([fert_id])[0]

        # Lookup quantity + sustainability note from SAME dataset
        row_match = df_master[df_master["Recommended_Fertilizer"] == fert_name].iloc[0]

        quantity = row_match["Quantity_kg_per_acre"]
        note = row_match["Sustainability_Note"]

        row = base.copy()
        row["Quantity_kg_per_acre"] = quantity
        row["Recommended_Fertilizer"] = fert_id

        X = pd.DataFrame([row])[feature_names]
        X_scaled = scaler.transform(X)

        yield_pred = reg.predict(X_scaled)[0]
        cost = yield_pred * 25000

        results.append({
            "Rank": rank,
            "Recommended Fertilizer": fert_name,
            "Quantity (kg/acre)": quantity,
            "Predicted Yield (ton/ha)": round(yield_pred, 2),
            "Confidence (%)": round(probs[fert_id] * 100, 2),
            "Sustainability Note": note
        })

    df = pd.DataFrame(results)

    print("\n🌾 LIVE SENSOR-BASED TOP-3 FERTILIZER RECOMMENDATIONS\n")
    print(df[[
        "Rank",
        "Recommended Fertilizer",
        "Quantity (kg/acre)",
        "Predicted Yield (ton/ha)",
        "Confidence (%)"
    ]].to_string(index=False))

    print("\n🌱 SUSTAINABILITY GUIDELINES")
    for _, r in df.iterrows():
        print(f"\n🔹 {r['Recommended Fertilizer']}:")
        print(r["Sustainability Note"])


# ================= SERIAL MONITOR =================

class SerialMonitor:
    def __init__(self):
        self.port = "COM3"
        self.baudrate = 115200
        self.ser = None
        self.sensor_data = {}

    def connect(self):
        print("🔌 Connecting to ESP32...")
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"--> Connected to {self.port}\n")
        except Exception:
            print("--> Failed to connect serial port")
            traceback.print_exc()
            sys.exit(1)

    def parse_line(self, line):
        try:
            if "DS18B20 Temperature" in line:
                self.sensor_data["soil_temp"] = float(line.split(":")[1].replace("°C", "").strip())

            elif "Soil Moisture Value" in line:
                raw = float(line.split(":")[1].strip())
                self.sensor_data["soil_moisture"] = (raw / 4095) * 100

            elif "DHT11 Temperature" in line:
                self.sensor_data["air_temp"] = float(line.split(":")[1].replace("°C", "").strip())

            elif "DHT11 Humidity" in line:
                self.sensor_data["air_humidity"] = float(line.split(":")[1].replace("%", "").strip())
        except:
            pass

    def read_once(self):
        print("--> Waiting for sensor data...\n")

        while True:
            raw = self.ser.readline()
            if not raw:
                continue

            try:
                line = raw.decode("utf-8").strip()
            except:
                continue

            print(f"--> {line}")
            self.parse_line(line)

            required = ["soil_temp", "soil_moisture", "air_temp", "air_humidity"]
            if all(k in self.sensor_data for k in required):
                return self.sensor_data


# ================= MAIN =================

if __name__ == "__main__":

    monitor = SerialMonitor()
    monitor.connect()

    sensor_data = monitor.read_once()

    print("\n📋 ENTER AGRONOMIC DETAILS")
    print("Paddy Growth Stages: Seedling, Tillering, Flowering, Panicle Initiation")
    growth_stage = input("Enter Paddy Growth Stage: ")

    print("\nPurpose Examples:")
    print("- Nitrogen and sulphur supply")
    print("- Phosphorus for root development")
    print("- Iron supply to prevent chlorosis")
    purpose = input("Enter Purpose: ")

    predict_top3(sensor_data, growth_stage, purpose)
