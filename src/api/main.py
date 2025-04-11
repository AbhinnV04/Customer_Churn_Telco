from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pandas as pd
import os
import logging

# Internal utils
from utils.model_loader import load_model, test_model_json
from utils.preprocessing import preprocessAPIData, get_trained_encoder_and_columns, 

# Load environment
load_dotenv()

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Module resolution (optional, if using relative imports elsewhere)
os.environ["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# FastAPI App
app = FastAPI()

# CORS middleware
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expected model input columns
MODEL_COLUMNS = [ 
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Load model
model = load_model()

# Input Schema
class CustomerData(BaseModel):
    gender: str
    seniorCitizen: int
    partner: int
    dependents: int
    tenure: float
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

    class Config:
        allow_population_by_field_name = True


# Utility: Rename keys to match training time casing
def rename_keys_for_model(d):
    key_map = {
        "gender": "gender",
        "seniorCitizen": "SeniorCitizen",
        "partner": "Partner",
        "dependents": "Dependents",
        "tenure": "tenure",
        "phoneService": "PhoneService",
        "multipleLines": "MultipleLines",
        "internetService": "InternetService",
        "onlineSecurity": "OnlineSecurity",
        "onlineBackup": "OnlineBackup",
        "deviceProtection": "DeviceProtection",
        "techSupport": "TechSupport",
        "streamingTV": "StreamingTV",
        "streamingMovies": "StreamingMovies",
        "contract": "Contract",
        "paperlessBilling": "PaperlessBilling",
        "paymentMethod": "PaymentMethod",
        "monthlyCharges": "MonthlyCharges",
        "totalCharges": "TotalCharges"
    }
    return {key_map.get(k, k): v for k, v in d.items()}


# Root Endpoint
@app.get("/")
async def root():
    logger.info("Root endpoint hit.")
    return {"message": "Welcome to the Churn Prediction API!"}


# Prediction Endpoint
@app.post("/predict")
async def predict_churn_new(data: CustomerData):
    try:
        data_dict = data.model_dump(by_alias=True)
        renamed = rename_keys_for_model(data_dict)

        input_df = pd.DataFrame([renamed])
        encoder, cat_cols, _ = get_trained_encoder_and_columns(input_df)

        data_preprocessed = preprocessAPIData(renamed, encoder, cat_cols, MODEL_COLUMNS)

        if data_preprocessed.shape[1] != len(MODEL_COLUMNS):
            raise ValueError(f"Input has {data_preprocessed.shape[1]} features, but model expects {len(MODEL_COLUMNS)}.")

        prediction = model.predict(data_preprocessed)[0]
        logger.info("✅ Prediction complete.")

        return {"churn_prediction": int(prediction)}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Legacy Predict Endpoint
@app.post("/predict_old")
def predict_churn_old(data: dict):
    logger.warning("Deprecated endpoint hit.")
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return {"churn_prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Test Endpoint
@app.get("/test")
def test_model_old():
    logger.info("Test model endpoint hit.")
    return test_model_json()
