import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib


def preprocess(df, save_artifacts=True, artifact_dir="models"):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Region", "Rice_Type", "Date"]).reset_index(drop=True)


    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df.fillna(df.median(numeric_only=True))

    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for t in ["Paddy_Price_LKR_per_kg", "Demand_Tons"]:
        if t in numeric_cols:
            numeric_cols.remove(t)

    mm_scaler = MinMaxScaler()
    df_mm = df.copy()
    if numeric_cols:
        df_mm[numeric_cols] = mm_scaler.fit_transform(df_mm[numeric_cols])

    std_scaler = StandardScaler()
    df_std = df.copy()
    if numeric_cols:
        df_std[numeric_cols] = std_scaler.fit_transform(df_std[numeric_cols])

    scalers = {"minmax": mm_scaler, "standard": std_scaler}
    artifacts = {"scalers": scalers}

    if save_artifacts:
        import os
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(scalers, f"{artifact_dir}/scalers.joblib")
        print(f"Saved artifacts to {artifact_dir}/")

    return df, df_mm, df_std, artifacts


if __name__ == "__main__":
    from load import load_dataset
    df = load_dataset()
    df_raw, df_mm, df_std, artifacts = preprocess(df, save_artifacts=False)
    print(df_mm.head())