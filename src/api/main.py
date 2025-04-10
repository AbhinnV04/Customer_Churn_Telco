from fastapi import FastAPI
import pandas as pd
import numpy as np 
from utils.model_loader import load_model, test_model_json
from utils.preprocessing import preprocessAPIData
from pydantic import BaseModel

class CustomerData(BaseModel):
    gender: str
    seniorCitizen: int
    partner: int
    dependents: int
    tenure: int 
    phoneService: int 
    multipleLines: str 
    internetService: str 
    onlineSecurity: str 
    onlineBackup: str 
    deviceProtection: str 
    techSupport: str 
    streamingTV: str 
    streamingMovies: str 
    contract: str 
    paperlessBilling: int 
    paymentMethod: str 
    monthlyCharges: float
    totalCharges: float    
    
app = FastAPI()
model = load_model()

@app.post("/predict")
async def predict_churn_new(data: CustomerData):
    # data_dict = data.model_dump()
    # data_df = pd.DataFrame([data_dict])
    # data_preprocessed = preprocessAPIData(data_df)
    # prediction = model.predict(data_preprocessed)[0] 
    return {"churn_prediction" : 1}
    
    
@app.post("/predict_old")
def predict_churn_old(data: dict):
    print("DEPRECATED")
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0] 
    return {"churn_prediction" : prediction}

@app.get("/test")
def test_model_old():
    return test_model_json()
