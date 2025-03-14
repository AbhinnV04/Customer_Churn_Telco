# src/utils/preprocessing.py
"""
Data cleaning and preprocessing
"""

import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

def load_data(FILENAME: str = 'Telco_customer_churn.csv') -> pd.DataFrame | None:
    """Loads data from the data directory"""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(curr_dir))
    data_path = os.path.join(project_root, "data", FILENAME)

    try:
        data = pd.read_csv(data_path)
        return data
    except Exception as e:
        print(f"{e}: Data file not found at {data_path}")
        return None

def preprocess_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Converts 'TotalCharges' to numeric and drops the rows with missing values."""
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    return df

def drop_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Drops the 'customerID' column."""
    df = df.drop('customerID', axis=1)
    return df

def one_hot_encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Performs one-hot encoding on categorical columns"""
    categorical_columns = df.select_dtypes(include=['boolean', 'object']).columns
    encoder = OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
    df = df.drop(columns=categorical_columns)
    df = df.reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def rename_churn_column(df: pd.DataFrame) -> pd.DataFrame:
    """Renames 'Churn_Yes' to 'Churn' if the column exists."""
    if 'Churn_Yes' in df.columns:
        df = df.rename(columns={"Churn_Yes": "Churn"})
    return df

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Combines all preprocessing steps into a single function."""
    df = preprocess_total_charges(df)
    df = drop_customer_id(df)
    df = one_hot_encode_categorical(df)
    df = rename_churn_column(df)
    return df

if __name__ == "__main__":
    # Test Data Creation
    # test_data = pd.DataFrame({
    #     'customerID': ['1234-ABCD', '5678-EFGH'],
    #     'TotalCharges': ['100.5', ' '],
    #     'gender': ['Male', 'Female'],
    #     'Churn_Yes': [1, 0],
    #     'Contract': ['Month-to-month', 'One year']
    # })

    # # Test load_data (basic check)
    # print("Testing load_data (basic check)...")
    # loaded_data = load_data()
    # # if loaded_data is not None:
    # #     print("load_data loaded successfully.")
    # # else:
    # #     print("load_data failed.")

    # # Test preprocess_total_charges
    # print("\nTesting preprocess_total_charges...")
    # processed_total_charges = preprocess_total_charges(test_data.copy())
    # # if len(processed_total_charges) == 1 and processed_total_charges['TotalCharges'].iloc[0] == 100.5:
    # #     print("preprocess_total_charges passed.")
    # # else:
    # #     print("preprocess_total_charges failed.")

    # # Test drop_customer_id
    # print("\nTesting drop_customer_id...")
    # dropped_id = drop_customer_id(test_data.copy())
    # if 'customerID' not in dropped_id.columns:
    #     print("drop_customer_id passed.")
    # else:
    #     print("drop_customer_id failed.")

    # Test one_hot_encode_categorical
    # print("\nTesting one_hot_encode_categorical...")
    # encoded_data = one_hot_encode_categorical(test_data.copy())
    # print(encoded_data)
    # if 'gender_Male' in encoded_data.columns and 'gender_Female' not in encoded_data.columns:
    #     print("one_hot_encode_categorical passed.")
    # else:
    #     print("one_hot_encode_categorical failed.")

    # # Test rename_churn_column
    # print("\nTesting rename_churn_column...")
    # renamed_data = rename_churn_column(test_data.copy())
    # if 'Churn' in renamed_data.columns and 'Churn_Yes' not in renamed_data.columns:
    #     print("rename_churn_column passed.")
    # else:
    #     print("rename_churn_column failed.")

    # # Test preprocess_dataframe
    # print("\nTesting preprocess_dataframe...")
    # preprocessed_data = preprocess_dataframe(test_data.copy())
    # if 'Churn' in preprocessed_data.columns and 'customerID' not in preprocessed_data.columns:
    #     print("preprocess_dataframe passed.")
    # else:
    #     print("preprocess_dataframe failed.")
    'Testing Testing hello'
    pass
    