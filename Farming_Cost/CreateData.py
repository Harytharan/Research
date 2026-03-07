import pandas as pd
import numpy as np
import random

# -----------------------------
# Seed for reproducibility
# -----------------------------
random.seed(42)
np.random.seed(42)

# -----------------------------
# Regions and GPS Coordinates
# -----------------------------
regions = {
    "Ampara": {"lat": (7.25, 7.45), "lon": (81.60, 81.80)},
    "Anuradhapura": {"lat": (8.25, 8.45), "lon": (80.35, 80.55)},
    "Polonnaruwa": {"lat": (7.90, 8.10), "lon": (80.90, 81.10)},
    "Kurunegala": {"lat": (7.35, 7.55), "lon": (80.25, 80.45)},
    "Jaffna": {"lat": (9.60, 9.80), "lon": (80.00, 80.25)},
    "Batticaloa": {"lat": (7.60, 7.85), "lon": (81.65, 81.85)},
}

soil_types = ["Clay", "Sandy Loam", "Loam", "Silt"]
seasons = ["Maha", "Yala"]
seed_types = ["BG300", "BG352", "BG358", "At306", "At402"]

# -----------------------------
# Helper function for region selection
# -----------------------------
def random_region():
    region = random.choice(list(regions.keys()))
    lat = round(random.uniform(*regions[region]["lat"]), 4)
    lon = round(random.uniform(*regions[region]["lon"]), 4)
    return region, lat, lon

# -----------------------------
# Generate synthetic data
# -----------------------------
data = []
for _ in range(218):
    region, lat, lon = random_region()
    year = random.choice(range(2017, 2026))
    season = random.choice(seasons)
    soil = random.choice(soil_types)
    rainfall = round(np.random.uniform(1000, 2500), 2)
    temp = round(np.random.uniform(26, 33), 2)
    humidity = round(np.random.uniform(70, 90), 2)
    area = round(np.random.uniform(0.5, 5.0), 2)
    seed_type = random.choice(seed_types)

    # Individual cost factors (in LKR)
    seed_cost = round(np.random.uniform(3000, 10000), 2)
    fertilizer_cost = round(np.random.uniform(8000, 20000), 2)
    pesticide_cost = round(np.random.uniform(2000, 6000), 2)
    labor_cost = round(np.random.uniform(15000, 35000), 2)
    water_cost = round(np.random.uniform(3000, 8000), 2)
    machinery_cost = round(np.random.uniform(7000, 20000), 2)
    other_costs = round(np.random.uniform(1000, 5000), 2)

    # Total cost (realistic weighted sum)
    total_cost = round(
        seed_cost
        + fertilizer_cost
        + pesticide_cost
        + labor_cost
        + water_cost
        + machinery_cost
        + other_costs
        + np.random.uniform(-1000, 1000), 2
    )

    yield_kg = round(np.random.uniform(1800, 4000) * area, 2)
    cost_per_kg = round(total_cost / yield_kg, 2)

    data.append([
        lat, lon, region, year, season, soil, rainfall, temp, humidity,
        area, seed_type, seed_cost, fertilizer_cost, pesticide_cost,
        labor_cost, water_cost, machinery_cost, other_costs,
        total_cost, yield_kg, cost_per_kg
    ])

# -----------------------------
# Create DataFrame
# -----------------------------
columns = [
    "Latitude", "Longitude", "Region", "Year", "Season", "Soil_Type",
    "Rainfall_mm", "Temperature_C", "Humidity_%",
    "Area_acres", "Seed_Type", "Seed_Cost (LKR)", "Fertilizer_Cost (LKR)",
    "Pesticide_Cost (LKR)", "Labor_Cost (LKR)", "Water_Cost (LKR)",
    "Machinery_Cost (LKR)", "Other_Costs (LKR)",
    "Total_Cost (LKR)", "Yield_kg", "Cost_per_kg (LKR)"
]

df = pd.DataFrame(data, columns=columns)

# -----------------------------
# Save to CSV
# -----------------------------
output_file = "paddy_farming_cost_dataset.csv"
df.to_csv(output_file, index=False)

print(f"✅ Dataset created successfully with {len(df)} rows!")
print(f"📁 Saved as: {output_file}")
print("\n📊 Preview:")
print(df.head(10))
