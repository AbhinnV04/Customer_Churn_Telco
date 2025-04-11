import joblib
import os
import numpy as np
import warnings  

warnings.simplefilter("ignore")  

_model = None

def load_model():
    """Loads and returns the trained model (lazy-loaded)."""
    global _model
    if _model is None:  # Load only if not already loaded
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "../../models/v1.2_Gradient_Boosting_calibrated.pkl"))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        _model = joblib.load(model_path)
    return _model


def test_model():
    """Tests the model with a sample input."""
    model = load_model()  
    
    sample_data = np.array([[0, 28, 104.8, 3046.05, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 
                             1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                             0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 
                             0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])  # Expected class: 1

    prediction = model.predict(sample_data)

    print("Expected Class: 1.0")
    print("Predicted Class:", prediction[0])
    
    
def test_model_json():
    """Tests the model with a sample input."""
    model = load_model()  
    
    sample_data = np.array([[0, 28, 104.8, 3046.05, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 
                             1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                             0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 
                             0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])  # Expected class: 1

    prediction = model.predict(sample_data)

    return {
        "Expected Class": 1.0, 
        "Predicted Class":prediction[0]
        }

if __name__ == "__main__":
    test_model()
