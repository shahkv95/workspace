import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from utils.utils import save_model_weights

logging.basicConfig(level=logging.INFO)


def fit_random_forest_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> None:
    """
        Fit random forest classifier on the training set, make predictions on the test set, and print evaluation metrics.

        Args:
        X_train (np.ndarray): Training input features.
        X_test (np.ndarray): Test input features.
        y_train (np.ndarray): Training target labels.
        y_test (np.ndarray): Test target labels.
        model_name (str): Name of the model to be saved.

    Returns:
        None"""

    try:
        logging.info("\n================    RANDOM FOREST MODEL   ================\n")
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        logging.info(f"\nAccuracy: {accuracy:.2%}")

        cm = confusion_matrix(y_test, predictions)
        logging.info(f"\nConfusion Matrix:\n{cm}")

        report = classification_report(y_test, predictions)
        logging.info(f"\nClassification Report:\n{report}")

        # Save model weights
        save_model_weights(model, model_name, __file__)

        feature_importances = pd.DataFrame(
            model.feature_importances_,
            index=pd.DataFrame(X_train).columns,
            columns=["importance"],
        ).sort_values("importance", ascending=False)

        logging.info(
            "\n========================    IMPORTANT_FEATURES    ========================"
        )
        logging.info(feature_importances)

    except Exception as e:
        logging.error("\nError occurred while fitting the random forest model.")
        logging.exception(e)
