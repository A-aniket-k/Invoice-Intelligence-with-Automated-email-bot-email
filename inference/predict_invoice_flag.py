import joblib
import pandas as pd

# 1. Update paths to point to the correct classification files
MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler_invoice.pkl"

def load_artifacts():
    """Load both the trained classifier and the matching scaler."""
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = joblib.load(f)
    return model, scaler

def predict_invoice_flag(input_data: dict) -> pd.DataFrame:
    """Predicts flag (0 or 1) for new vendor invoices using scaled features."""
    model, scaler = load_artifacts()
    input_df = pd.DataFrame(input_data)
    
    # Define the exact feature order used during model training
    features = ["invoice_quantity", "invoice_dollars", "Freight", "total_item_quantity", "total_item_dollars"]
    
    # Scale the input features before predicting
    X_scaled = scaler.transform(input_df[features])
    
    # Generate classification predictions (0 = Normal, 1 = Flagged)
    input_df['Predicted_Flag'] = model.predict(X_scaled)
    return input_df

# 2. This block is now properly indented!
if __name__ == "__main__":
    # Example sample data matching the 5 required features
    sample_data = {
        "invoice_quantity": [10, 500, 2],
        "invoice_dollars": [150.0, 25000.0, 30.0],
        "Freight": [15.0, 1200.0, 5.0],
        "total_item_quantity": [10, 480, 2],
        "total_item_dollars": [140.0, 24000.0, 28.0]
    }

    prediction = predict_invoice_flag(sample_data)
    print("\n--- Invoice Flagging Predictions ---")
    print(prediction)