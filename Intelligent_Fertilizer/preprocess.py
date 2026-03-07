import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path="fertilizer_recommendation_dataset.csv"):
    df = pd.read_csv(file_path)

    # Drop irrelevant columns
    drop_cols = ['Date', 'Sustainability_Note', 'Recommended_Fertilizer',
                 'Predicted_Yield_ton_per_ha', 'Cost_LKR_per_ha']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Separate target
    y = df['Fertilizer']
    X = df.drop(columns=['Fertilizer'])

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Save feature names for later
    feature_names = X.columns.tolist()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, target_encoder, label_encoders, feature_names
