# K - nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import Tuple


def fit_knn_model(X_train: Tuple, X_test: Tuple, y_train: Tuple, y_test: Tuple) -> None:
    """
    Fit k-nearest neighbors classifier on the training set, make predictions on the test set, and print evaluation metrics.
    """
    print("\n================    KNN MODEL    ================\n")
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2%}")

    cm = confusion_matrix(y_test, predictions)
    print(f"Confusion Matrix:\n{cm}")

    report = classification_report(y_test, predictions)
    print(f"Classification Report:\n{report}")
