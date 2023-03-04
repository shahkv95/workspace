# Random forest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import Tuple
import joblib
import os


def fit_random_forest_model(
    X_train: Tuple, X_test: Tuple, y_train: Tuple, y_test: Tuple, model_name: str
) -> None:
    """
    Fit random forest model on the training set, make predictions on the test set, and print evaluation metrics.
    """
    print("\n================    RANDOM FOREST MODEL   ================\n")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.2%}")

    cm = confusion_matrix(y_test, predictions)
    print(f"Confusion Matrix:\n{cm}")

    report = classification_report(y_test, predictions)
    print(f"Classification Report:\n{report}")

    # Save model weights
    model_weights_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_weights"
    )
    os.makedirs(model_weights_dir, exist_ok=True)
    weights_file_path = os.path.join(model_weights_dir, f"{model_name}_weights.joblib")
    joblib.dump(model, weights_file_path)

    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=pd.DataFrame(X_train).columns,
        columns=["importance"],
    ).sort_values("importance", ascending=False)

    print("========================    IMPORTANT_FEATURES    ========================")
    print(feature_importances)
