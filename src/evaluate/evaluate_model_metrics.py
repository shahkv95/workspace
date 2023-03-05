import logging
from typing import Any
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO)


def evaluate_model_metrics(y_test: np.ndarray, predictions: np.ndarray) -> None:
    """

    Calculate and log evaluation metrics for a model.

    Args:
        y_test (np.ndarray): The true labels.
        predictions (np.ndarray): The predicted labels.

    Returns:
        None
    """

    try:
        logging.info("Evaluating model metrics....")
        accuracy = accuracy_score(y_test, predictions)
        logging.info(f"Accuracy: {accuracy:.2%}\n")

        cm = confusion_matrix(y_test, predictions)
        logging.info(f"Confusion Matrix:\n{cm}\n")

        report = classification_report(y_test, predictions)
        logging.info(f"Classification Report:\n{report}\n")

    except Exception as e:
        logging.error("An error occurred while evaluating the model metrics.")
        logging.exception(e)
