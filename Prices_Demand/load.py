import pandas as pd

def load_dataset(path="paddy_price_demand_dataset.csv"):

    df = pd.read_csv(path)
    # Ensure Date is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    else:
        raise ValueError("Dataset must contain a 'Date' column.")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
