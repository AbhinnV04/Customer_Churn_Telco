from fastapi import FastAPI
import pandas as pd
import numpy as np 
from utils.model_loader import load_model, test_model_json

app = FastAPI()
model = load_model()

@app.post("/predict")
def predict_churn(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    
    return {"churn_prediction" : prediction}

@app.get("/test")
def test_model():
    return test_model_json()
