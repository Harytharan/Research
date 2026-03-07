import joblib
import pandas as pd
import os


# ---------- LOAD MASTER DATASET ----------
MASTER_DATA = "Dataset.csv"
df_master = pd.read_csv(MASTER_DATA)


def predict_top3(
    soil_temp,
    soil_moisture,
    air_temp,
    air_humidity,
    paddy_growth_stage,
    purpose
):
    model_dir = ""

    # Load models
    clf = joblib.load(os.path.join(model_dir, "fertilizer_models/fertilizer_model.pkl"))
    reg = joblib.load(os.path.join(model_dir, "yield_models/yield_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "yield_models/scaler.pkl"))
    encoders = joblib.load(os.path.join(model_dir, "fertilizer_models/label_encoders.pkl"))
    feature_names = joblib.load(os.path.join(model_dir, "fertilizer_models/feature_names.pkl"))

    # Base input
    base = {
        "Soil_Temperature (°C)": soil_temp,
        "Soil_Moisture (%)": soil_moisture,
        "Air_Temperature (°C)": air_temp,
        "Air_Humidity (%)": air_humidity,
        "Paddy_Growth_Stage": encoders["Paddy_Growth_Stage"].transform([paddy_growth_stage])[0],
        "Purpose": encoders["Purpose"].transform([purpose])[0],
        "Quantity_kg_per_acre": 25   # default (will override from dataset)
    }

    # Dummy fertilizer for probability calculation
    X_base = pd.DataFrame([{**base, "Recommended_Fertilizer": 0}])[feature_names]
    probs = clf.predict_proba(scaler.transform(X_base))[0]

    top3 = probs.argsort()[-3:][::-1]

    results = []

    for rank, fert_id in enumerate(top3, start=1):
        fert_name = encoders["Recommended_Fertilizer"].inverse_transform([fert_id])[0]

        # ---- LOOKUP quantity + sustainability note ----
        match = df_master[df_master["Recommended_Fertilizer"] == fert_name].iloc[0]

        quantity = match["Quantity_kg_per_acre"]
        sustainability_note = match["Sustainability_Note"]

        # Predict yield
        temp = base.copy()
        temp["Quantity_kg_per_acre"] = quantity
        temp["Recommended_Fertilizer"] = fert_id

        X = pd.DataFrame([temp])[feature_names]
        X_scaled = scaler.transform(X)

        yield_pred = reg.predict(X_scaled)[0]
        cost = yield_pred * 25000

        results.append({
            "Rank": rank,
            "Recommended Fertilizer": fert_name,
            "Quantity (kg/acre)": quantity,
            "Predicted Yield (ton/ha)": round(yield_pred, 2),
            "Estimated Cost (LKR/ha)": round(cost, 2),
            "Confidence (%)": round(probs[fert_id] * 100, 2),
            "Sustainability Note": sustainability_note
        })

    df = pd.DataFrame(results)

    print("\n🌾 TOP-3 FERTILIZER RECOMMENDATIONS\n")
    print(df[[
        "Rank",
        "Recommended Fertilizer",
        "Quantity (kg/acre)",
        "Predicted Yield (ton/ha)",
        "Confidence (%)"
    ]].to_string(index=False))

    print("\n🌱 SUSTAINABILITY GUIDELINES")
    for _, row in df.iterrows():
        print(f"\n🔹 {row['Recommended Fertilizer']}:")
        print(row["Sustainability Note"])

    return df


# ---------- CLI INPUT ----------
if __name__ == "__main__":

    print("\n🌱 Enter Paddy Field Details\n")

    soil_temp = float(input("Soil Temperature (°C): "))
    soil_moisture = float(input("Soil Moisture (%): "))
    air_temp = float(input("Air Temperature (°C): "))
    air_humidity = float(input("Air Humidity (%): "))

    print("\nPaddy Growth Stages: Seedling, Tillering, Flowering, Maturity")
    paddy_growth_stage = input("Enter Paddy Growth Stage: ")

    print("\nPurpose Examples:")
    print("- Nitrogen for leaf growth")
    print("- Phosphorus for root development")
    print("- Potassium for grain filling")
    purpose = input("Enter Purpose: ")

    predict_top3(
        soil_temp,
        soil_moisture,
        air_temp,
        air_humidity,
        paddy_growth_stage,
        purpose
    )
