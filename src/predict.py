import os
import pandas as pd
import joblib
from src.preprocess import preprocess_data


# Load model once (best practice)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))


def predict(input_data: dict):
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # Preprocess data (no training)
    X_scaled, _ = preprocess_data(df, training=False)

    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    return prediction, probability
