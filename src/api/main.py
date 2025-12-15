from fastapi import FastAPI
from pydantic_models import CustomerRFM
import pandas as pd
from predict import load_model, predict_risk

app = FastAPI()
model = load_model()

@app.post("/predict")
def predict(customer: CustomerRFM):
    input_df = pd.DataFrame([customer.dict()])
    risk_prob = predict_risk(model, input_df)
    return {"risk_probability": float(risk_prob[0])}
