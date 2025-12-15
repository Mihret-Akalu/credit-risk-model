from fastapi import FastAPI
from pydantic_models import CustomerData
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('../models/best_model.pkl')

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[:, 1][0]
    return {"risk_probability": prob}
