# src/api/main.py
from fastapi import FastAPI
from pydantic_models import CustomerData
import pandas as pd
import joblib
from predict import predict_risk

app = FastAPI(title="Credit Risk API")

# Load model
model = joblib.load("models/credit_risk_model.pkl")

@app.post("/predict")
def predict(customer: CustomerData):
    input_df = pd.DataFrame([customer.dict()])
    risk_prob = predict_risk(model, input_df)[0]
    return {"risk_probability": risk_prob}
