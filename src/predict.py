import pandas as pd
import joblib

model = joblib.load('../models/best_model.pkl')

def predict(df):
    return model.predict_proba(df)[:, 1]
