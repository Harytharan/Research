
def create_features(df):
    target = "Total_Cost (LKR)"
    features = [
        "Latitude", "Longitude", "Year", "Season", "Soil_Type",
        "Rainfall_mm", "Temperature_C", "Humidity_%",
        "Area_acres", "Seed_Type", "Seed_Cost (LKR)", "Fertilizer_Cost (LKR)",
        "Pesticide_Cost (LKR)", "Labor_Cost (LKR)", "Water_Cost (LKR)",
        "Machinery_Cost (LKR)", "Other_Costs (LKR)"
    ]

    X = df[features]
    y = df[target]

    print(f"Feature Engineering complete: {X.shape[1]} features selected.")
    return X, y

if __name__ == "__main__":
    from load import load_dataset
    from pre_process import preprocess_data
    df = load_dataset()
    df, _ = preprocess_data(df)
    X, y = create_features(df)
    print(X.head())
