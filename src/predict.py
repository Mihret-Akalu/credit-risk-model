# src/predict.py
import pandas as pd
import joblib

def load_model(model_path="models/credit_risk_model.pkl"):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def predict_risk(model, input_df):
    """
    Predict credit risk probability
    """
    return model.predict_proba(input_df)[:,1]
