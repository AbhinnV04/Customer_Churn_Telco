# src/utils/preprocessing.py
"""
Data cleaning and preprocessing
"""
import pandas as pd 

def preprocessAPIData(data: pd.DataFrame):
    data = data.copy()
    return data

if __name__ == "__main__":
    data = pd.DataFrame()
    print(preprocessAPIData(data))
    pass