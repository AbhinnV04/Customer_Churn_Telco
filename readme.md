# Customer Churn Prediction Project

![](https://trianglerice.wordpress.com/wp-content/uploads/2015/01/tumblr_nhk406ny5e1ti2my5o1_5001.gif)

This project focuses on predicting customer churn using machine learning techniques. It includes data exploration, model building, and a FastAPI API for serving predictions.

## Description

The project is organized as follows:

- **`models/`**: Contains the saved trained model pipeline (`model.joblib`).
- **`data/`**: Stores the raw and (optionally) processed data.
- **`notebooks/`**: Includes Jupyter notebooks used for data exploration, model development, and experimentation.
- **`src/`**: Contains the source code:
  - **`api/`**: Implements the FastAPI application for serving predictions.
  - **`utils/`**: Contains utility functions, including:
    - `model_builder.py`: Script for rebuilding and training the model.
    - `preprocessing.py`: Functions for data preprocessing (if needed outside the pipeline).
- **`tests/`**: Contains tests for the API and other components.
- **`requirements.txt`**: Lists the Python dependencies.
- **`README.md`**: Provides project documentation.
- **`.gitignore`**: Specifies files to be ignored by Git.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip`
- Git (optional, for version control)

### Installation

1. **Clone the repository (if using Git):**

    ```bash
    git clone https://github.com/AbhinnV04/Customer_Churn_Telco.git
    cd <project_directory>
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    ```

    - **Activate the virtual environment:**
      - **Windows:**
        ```bash
        venv\Scripts\activate
        ```
      - **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Rebuilding and Training the Model

To rebuild and train the model, run the `model_builder.py` script:

```bash
python src/utils/model_builder.py
```
- This will train the model and save the pipeline to models/model.joblib.

### Running the FastAPI API
To start the FastAPI server:
```
cd src/api
uvicorn main:app --reload
```

### Project Components
#### 1. Model Building
The model is built and trained using scikit-learn.
The pipeline includes preprocessing steps and a Decision Tree classifier (potentially with calibration).
#### 2. API
The FastAPI application provides endpoints for making predictions.
It loads the trained model pipeline and uses it to generate predictions from input data.
#### 3. Data Handling
The project assumes data is stored in CSV files, but it can be adapted to other formats.
#### 4. Feature Importance
Feature importances are extracted from the Decision Tree model to understand the key drivers of customer churn.
#### 5. Calibration
Model calibration is used to improve the reliability of predicted probabilities.

---

<div style="text-align: center;">
    <img src="https://media2.giphy.com/media/a6pzK009rlCak/200.gif?cid=6c09b952zrtwcjwop83jxm86d7cin1pwqbwivmg6jej9dngs&ep=v1_gifs_search&rid=200.gif&ct=g" alt="Alternative text">
    <p>*Arigato for Reading this far*</p>
</div>

---
