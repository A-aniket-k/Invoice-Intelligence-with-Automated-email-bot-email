import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data: dict) -> pd.DataFrame:
    """
    Predict freight cost for new vendor invoices.
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    
    # Check if data came from the UI ("Invoice Dollars") or local test ("Dollars")
    if "Invoice Dollars" in input_df.columns:
        target_col = "Invoice Dollars"
    elif "Dollars" in input_df.columns:
        target_col = "Dollars"
    else:
        raise KeyError("Could not find a valid dollar amount column ('Invoice Dollars' or 'Dollars') in input data.")
        
    # Create the exact single-column feature DataFrame with the name expected by the model
    X_predict = pd.DataFrame({"Dollars": input_df[target_col]})
    
    # Generate predictions and round them
    input_df['Predicted_Freight'] = model.predict(X_predict).round()
    return input_df

if __name__ == "__main__":
    # Example inference run (local testing still works perfectly!)
    sample_data = {
        "Dollars": [18500, 9000, 3000, 200]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)