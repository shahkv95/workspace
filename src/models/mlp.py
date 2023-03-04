# MLP Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import Tuple


def fit_mlp_classifier_model(
    X_train: Tuple, X_test: Tuple, y_train: Tuple, y_test: Tuple
) -> None:
    """
    Fit MLP classifier model on the training set, make predictions on the test set, and print evaluation metrics.
    """

    print("\n================    MLP MODEL    ================\n")
    model = MLPClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2%}")

    cm = confusion_matrix(y_test, predictions)
    print(f"Confusion Matrix:\n{cm}")

    report = classification_report(y_test, predictions)
    print(f"Classification Report:\n{report}")
