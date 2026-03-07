import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ---- Parameters ----
num_rows = 5000
start_date = datetime(2024, 1, 1)

# ---- Possible values ----
paddy_stages = ["Seedling", "Tillering", "Panicle Initiation", "Flowering", "Maturity"]
pests = [
    "Brown Planthopper", "Green Leafhopper", "Rice Stem Borer",
    "Rice Gall Midge", "Rice Leaf Folder", "Rice Blast", "Armyworm", "Rice Hispa"
]
risk_levels = ["Low", "Moderate", "High"]
actions = {
    "Brown Planthopper": "Use resistant varieties; apply imidacloprid if severe.",
    "Green Leafhopper": "Maintain field hygiene; use neem-based sprays.",
    "Rice Stem Borer": "Use pheromone traps; release Trichogramma.",
    "Rice Gall Midge": "Avoid excess nitrogen; apply chlorpyrifos if needed.",
    "Rice Leaf Folder": "Use light traps; spray chlorantraniliprole for control.",
    "Rice Blast": "Ensure proper spacing; apply tricyclazole if infected.",
    "Armyworm": "Monitor at night; use biological control or spinosad.",
    "Rice Hispa": "Manual removal of affected leaves; apply neem extract."
}

# ---- Generate synthetic but realistic sensor data ----
data = []
for i in range(num_rows):
    date = start_date + timedelta(days=i)
    
    temp = round(np.random.normal(30, 3), 1)  # temperature in °C
    humidity = round(np.random.uniform(60, 95), 1)
    pressure = round(np.random.normal(1008, 5), 1)
    light_intensity = round(np.random.uniform(10000, 70000), 0)
    
    stage = random.choice(paddy_stages)
    
    # Random pest probability based on temperature/humidity
    if temp > 33 and humidity > 80:
        pest = random.choice(["Brown Planthopper", "Green Leafhopper"])
    elif temp < 27 and humidity < 70:
        pest = random.choice(["Rice Blast", "Rice Leaf Folder"])
    elif humidity > 90:
        pest = random.choice(["Rice Gall Midge", "Armyworm"])
    else:
        pest = random.choice(pests)
    
    risk = random.choices(risk_levels, weights=[0.4, 0.4, 0.2])[0]
    action = actions[pest]
    
    data.append([
        date.strftime("%Y-%m-%d"), temp, humidity, pressure, light_intensity,
        stage, pest, risk, action
    ])

# ---- Create DataFrame ----
columns = [
    "Date", "Temperature (°C)", "Humidity (%)", "Pressure (hPa)", "Light Intensity (lux)",
    "Paddy Stage", "Detected Pest", "Outbreak Risk Level", "Recommended Action"
]

df = pd.DataFrame(data, columns=columns)

# ---- Save to CSV ----
df.to_csv("paddy_pest_dataset.csv", index=False)
print("✅ Dataset created successfully: paddy_pest_dataset.csv")
print(df.head(10))
