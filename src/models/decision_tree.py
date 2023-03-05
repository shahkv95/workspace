import logging
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from evaluate.evaluate_model_metrics import evaluate_model_metrics

from utils.utils import save_model_weights

logging.basicConfig(level=logging.INFO)


def fit_decision_tree_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> None:
    """
    Fit decision tree classifier on the training set, make predictions on the test set, and print evaluation metrics.

    Args:
        X_train (np.ndarray): Training input features.
        X_test (np.ndarray): Test input features.
        y_train (np.ndarray): Training target labels.
        y_test (np.ndarray): Test target labels.
        model_name (str): Name of the model to be saved.

    Returns:
        None
    """
    try:
        logging.info("\n============= DECISION TREE MODEL =============\n")

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Calculate and log evaluation metrics for a model.
        evaluate_model_metrics(y_test, predictions)

        # Save model weights
        save_model_weights(model, model_name, __file__)

    except Exception as e:
        logging.error("\nError occurred while fitting the decision tree model.")
        logging.exception(e)
