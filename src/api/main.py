from fastapi import FastAPI
from .pydantic_models import CustomerData
import joblib
import pandas as pd

app = FastAPI()
import os
import joblib

# Get absolute path relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best_model.pkl')
model = joblib.load(MODEL_PATH)


@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[:, 1][0]
    return {"risk_probability": prob}
