import logging
import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


logging.basicConfig(level=logging.INFO)


def fit_knn_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> None:
    """
    Fit k-nearest neighbors classifier on the training set, make predictions on the test set, and print evaluation metrics.

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
        logging.info("\n================    KNN MODEL    ================\n")
        model = KNeighborsClassifier(n_neighbors=2)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        logging.info(f"Accuracy: {accuracy:.2%}")

        cm = confusion_matrix(y_test, predictions)
        logging.info(f"Confusion Matrix:\n{cm}")

        report = classification_report(y_test, predictions)
        logging.info(f"Classification Report:\n{report}")

        # Save model weights
        model_weights_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "model_weights"
        )
        os.makedirs(model_weights_dir, exist_ok=True)
        weights_file_path = os.path.join(
            model_weights_dir, f"{model_name}_weights.joblib"
        )
        joblib.dump(model, weights_file_path)

    except Exception as e:
        logging.error("\nError occurred while fitting the KNN model.")
        logging.exception(e)
