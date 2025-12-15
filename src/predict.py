import joblib
import pandas as pd

def load_model(path="../models/rf_model.pkl"):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_risk(model, input_df: pd.DataFrame):
    try:
        preds = model.predict_proba(input_df)[:,1]  # probability of high-risk
        return preds
    except Exception as e:
        print(f"Prediction error: {e}")
        return None
