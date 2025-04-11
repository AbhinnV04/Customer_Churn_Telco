# src/utils/preprocessing.py
"""
Data cleaning and preprocessing
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load this once at app startup after training
def get_trained_encoder_and_columns(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    encoder = OneHotEncoder(drop="if_binary", sparse_output=False)
    encoder.fit(df[cat_cols])
    encoded_columns = encoder.get_feature_names_out(cat_cols).tolist()
    return encoder, cat_cols, encoded_columns

# Inference-time preprocessor
def preprocess_API_data(data_dict, encoder, cat_cols, model_columns):
    df = pd.DataFrame([data_dict])
    encoded = encoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    df.drop(columns=cat_cols, inplace=True)
    df_final = pd.concat([df, encoded_df], axis=1)

    # Match columns
    df_final = df_final.reindex(columns=model_columns, fill_value=0)
    return df_final

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

def get_model_columns()-> list:
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
    return MODEL_COLUMNS


if __name__ == "__main__":
    data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "tenure": 12,
        "MonthlyCharges": 10000,
        "TotalCharges": 120000,
        "MultipleLines": "No",
        "InternetService": "DSL",
        "Contract": "One year",
        "PaymentMethod": "Bank transfer (automatic)",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No"
    }

    model_columns = get_model_columns()
    encoder, cat_cols, encoded_columns = get_trained_encoder_and_columns(pd.DataFrame([data]))
    data = preprocess_API_data(data, encoder, cat_cols, model_columns)
    print(data.T)
