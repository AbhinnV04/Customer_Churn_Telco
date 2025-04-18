# Customer Churn Prediction Project

<p align="center">
  <img src="https://abhinnv04.github.io/portfolio/assets/project_imgs/churn.gif" alt="Centered GIF" />
</p>


This project focuses on predicting customer churn using machine learning techniques. It includes data exploration, model building, and a FastAPI API for serving predictions.

## Description

The project is organized as follows:

- **`models/`**: Contains the saved trained model pipeline (`model.joblib`).
- **`data/`**: Stores the raw and (optionally) processed data.
- **`notebooks/`**: Includes Jupyter notebooks used for data exploration, model development, and experimentation.
- **`src/`**: Contains the source code:
  - **`api/`**: Implements the FastAPI application for serving predictions.
    - `main.py`: FastAPI app.
  - **`schemas/`**: Pydantic schema model for POST requests.
    - `customer.py`: Schema for customer.
  - **`utils/`**: Contains utility functions, including:
    - `model_loader.py`: Script for loading the latest model.
    - `preprocessing.py`: Functions for data preprocessing from the POST request for the model.
- **`requirements.txt`**: Lists the Python dependencies.
- **`example.env`**: Format for the actual .env file.
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

### Running the FastAPI API
To start the FastAPI server:
```
fastapi dev main.py
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
## 🌐 Deployment

The application is deployed and live!
You can interact with the app by entering customer details via the frontend, which then communicates with the deployed FastAPI backend for predictions.

- **Frontend (Vercel):**  
  🔗 [https://telco-churn.vercel.app/](https://telco-churn.vercel.app/)

- **Backend API (Render):**  
  🔗 [https://customer-churn-telco.onrender.com/](https://customer-churn-telco.onrender.com/)

- **Backend API Docs:**  
  🔗 [https://customer-churn-telco.onrender.com/docs](https://customer-churn-telco.onrender.com/docs)


---
