import joblib
from pathlib import Path
from data_preprocessing import (
    apply_labels,
    load_invoice_data,
    scale_features,
    split_data,
)
from modeling_evaluation import evaluate_classifier, train_random_forest

# Constants
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

TARGET = "flag_invoice"


def main():
    # Create models directory
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # Prepare data
    X_train, X_test, Y_train, Y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, str(model_dir / "scaler_invoice.pkl")
    )

    # Train and evaluate models
    grid_search = train_random_forest(X_train_scaled, Y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        Y_test,
        "Random Forest Classifier",
    )

    # Save best model
    joblib.dump(grid_search.best_estimator_, str(model_dir / "predict_flag_invoice.pkl"))


if __name__ == "__main__":
    main()