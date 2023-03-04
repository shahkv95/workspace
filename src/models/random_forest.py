# Random forest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import Tuple


def fit_random_forest_model(
    X_train: Tuple, X_test: Tuple, y_train: Tuple, y_test: Tuple
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

    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=pd.DataFrame(X_train).columns,
        columns=["importance"],
    ).sort_values("importance", ascending=False)

    print("========================    IMPORTANT_FEATURES    ========================")
    print(feature_importances)
