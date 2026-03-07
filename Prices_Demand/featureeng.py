import numpy as np

def create_lstm_sequences(df_mm, feature_cols, target_col="Paddy_Price_LKR_per_kg", window_size=14):

    values = df_mm[feature_cols].values
    target = df_mm[target_col].values
    X, y = [], []
    for i in range(window_size, len(df_mm)):
        X.append(values[i-window_size:i])
        y.append(target[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

def create_tabular_dataset(df_std, feature_cols, target_col="Demand_Tons"):

    X = df_std[feature_cols].values
    y = df_std[target_col].values
    return X, y

if __name__ == "__main__":
    import pandas as pd
    from load import load_dataset
    from pre_process import preprocess
    df = load_dataset()
    df_raw, df_mm, df_std, _ = preprocess(df, save_artifacts=False)
    # choose features
    feature_cols = [c for c in df_mm.columns if c not in ["Date", "Paddy_Price_LKR_per_kg", "Demand_Tons"]]
    X_seq, y_seq = create_lstm_sequences(df_mm, feature_cols, window_size=14)
    print("LSTM sequences:", X_seq.shape, y_seq.shape)
