from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pandas as pd
import os
import logging

# Internal utils
from utils.model_loader import load_model, test_model_json
from utils.preprocessing import preprocess_API_data, get_trained_encoder_and_columns, rename_keys_for_model, get_model_columns
from schemas.customer import CustomerData

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
MODEL_COLUMNS = get_model_columns()

# Load model
model = load_model()

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

        data_preprocessed = preprocess_API_data(renamed, encoder, cat_cols, MODEL_COLUMNS)

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
