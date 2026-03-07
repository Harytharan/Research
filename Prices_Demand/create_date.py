import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# ================= CONFIGURATION =================
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
REGIONS = ["North", "South", "East", "West", "Central"]
OUTPUT_DIR = "data"
OUTPUT_FILE = "paddy_market_data.csv"

random.seed(42)
np.random.seed(42)
# =================================================


def generate_data():
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")

    rows = []
    current_date = start

    while current_date <= end:
        for region in REGIONS:

            # -------- WEATHER --------
            rainfall_mm = round(np.random.gamma(shape=2.0, scale=15.0), 2)  # mm
            temperature_c = round(np.random.normal(27, 2.5), 2)            # Celsius

            # -------- SENTIMENT --------
            sentiment_score = round(np.random.uniform(-1, 1), 3)
            news_sentiment = round(np.random.uniform(-1, 1), 3)

            # -------- SOIL NUTRIENTS (NPK) --------
            nitrogen_n = round(random.uniform(450, 600), 1)    # mg/kg
            phosphorus_p = round(random.uniform(5, 15), 1)    # mg/kg
            potassium_k = round(random.uniform(5, 15), 1)     # mg/kg

            # -------- SEASONAL EFFECT --------
            month = current_date.month
            seasonal_factor = 1.0
            if month in [4, 5, 10, 11]:   # Cultivation seasons
                seasonal_factor = 1.15
            elif month in [1, 2]:
                seasonal_factor = 0.9

            # -------- PRICE MODEL --------
            base_price = 95
            price = (
                base_price
                + rainfall_mm * 0.03
                - temperature_c * 0.4
                + sentiment_score * 4
                + news_sentiment * 3
                + (nitrogen_n - 520) * 0.01
                + (phosphorus_p - 10) * 0.6
                + (potassium_k - 10) * 0.5
            ) * seasonal_factor

            price = round(max(price, 60), 2)  # Floor price

            # -------- DEMAND MODEL --------
            base_demand = 420
            demand = (
                base_demand
                + rainfall_mm * 0.4
                + sentiment_score * 30
                + news_sentiment * 25
                - price * 1.1
            ) * seasonal_factor

            demand = round(max(demand, 100), 2)

            # -------- APPEND ROW --------
            rows.append([
                current_date.strftime("%Y-%m-%d"),
                region,
                rainfall_mm,
                temperature_c,
                sentiment_score,
                news_sentiment,
                nitrogen_n,
                phosphorus_p,
                potassium_k,
                price,
                demand
            ])

        current_date += timedelta(days=1)

    return pd.DataFrame(rows, columns=[
        "Date",
        "Region",
        "Rainfall_mm",
        "Temperature_C",
        "Sentiment_Score",
        "News_Sentiment",
        "Nitrogen_N",
        "Phosphorus_P",
        "Potassium_K",
        "Paddy_Price_LKR_per_kg",
        "Demand_Tons"
    ])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = generate_data()
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    df.to_csv(output_path, index=False)

    print("✅ Dataset created successfully!")
    print(f"📁 File saved to: {output_path}")
    print(f"📊 Total rows: {len(df)}")
    print("\n🔍 Sample:")
    print(df.head())


if __name__ == "__main__":
    main()
