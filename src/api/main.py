from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pandas as pd
import os
import logging

# Internal utils
from utils.model_loader import load_model, test_model_json
from utils.preprocessing import preprocessAPIData

load_dotenv()

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Module Resolution
os.environ["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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

# Initialize model and App
app = FastAPI()
model = load_model()

origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins= origins,  # your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints

@app.get("/")
async def root():
    logger.info("YAY!! API is running")
    
    return {"message": "Welcome to the API!"}

@app.post("/predict")
async def predict_churn_new(data: CustomerData):
    # data_dict = data.model_dump()
    # data_df = pd.DataFrame([data_dict])
    # data_preprocessed = preprocessAPIData(data_df)
    # prediction = model.predict(data_preprocessed)[0] 
    return {"churn_prediction" : 1}
    
@app.post("/predict_old")
def predict_churn_old(data: dict):
    logger.info("DEPRECATED")
    
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0] 
    return {"churn_prediction" : prediction}

@app.get("/test")
def test_model_old():
    logger.info("Testing model...")
    
    result = test_model_json()
    return result
