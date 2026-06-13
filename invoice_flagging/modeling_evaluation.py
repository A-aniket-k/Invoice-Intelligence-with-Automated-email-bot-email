from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV


def train_random_forest(X_train, Y_train):
    """Tunes and trains a Random Forest Classifier using GridSearchCV."""
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"],
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, scoring=scorer, cv=5, n_jobs=-1, verbose=0
    )

    grid_search.fit(X_train, Y_train)
    return grid_search


def evaluate_classifier(model, X_test, Y_test, model_name):
    """Evaluates the model performance and prints accuracy and classification report."""
    preds = model.predict(X_test)

    accuracy = accuracy_score(Y_test, preds)
    report = classification_report(Y_test, preds)

    print(f"\n=== {model_name} Performance ===")
    print(f"Accuracy: {accuracy:.2f}")
    print("-" * 30)
    print(report)