import pandas as pd

def load_dataset(path="paddy_farming_cost_dataset.csv"):

    df = pd.read_csv(path)
    print("Dataset loaded successfully!")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nPreview:\n", df.head(5))
    return df

if __name__ == "__main__":
    df = load_dataset()
