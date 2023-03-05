import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)
from typing import Tuple


def fit_all_models(
    X_train: Tuple, X_test: Tuple, y_train: Tuple, y_test: Tuple
) -> None:
    print("\n================    ALL MODELS    ================\n")
    models = {
        "SVM": SVC(gamma=2, C=1),
        "KNN": KNeighborsClassifier(2),
        "Decision Tree": DecisionTreeClassifier(),
        "Adaboost": AdaBoostClassifier(),
        "Naive Bayes": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "MLP Classifier": MLPClassifier(),
        "Logistic Regression": LogisticRegression(solver="saga"),
        "Random Forest": RandomForestClassifier(),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        cr = classification_report(y_test, predictions)
        results[name] = {
            "Accuracy": accuracy,
            "F1-score": f1,
            "Confusion Matrix": cm,
            "Classification Report": cr,
        }
        if name == "Random Forest":
            feature_importances = pd.DataFrame(
                model.feature_importances_,
                index=pd.DataFrame(X_train).columns,
                columns=["importance"],
            ).sort_values("importance", ascending=False)
            print("\nRandom Forest Feature Importances:\n", feature_importances)

    print("\n")
    print("-" * 66)
    print(
        "{:<20s}{:<15s}{:<15s}{:<15s}{:<15s}".format(
            "Model", "Accuracy", "Precision", "Recall", "F1-score"
        )
    )
    print("-" * 66)
    for name, result in results.items():
        cm = result["Confusion Matrix"]
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        print(
            "{:<20s}{:<15.2f}{:<15.2f}{:<15.2f}{:<15.2f}".format(
                name, result["Accuracy"] * 100, precision, recall, result["F1-score"]
            )
        )
    print("-" * 66)

    # Create a line chart for accuracy, f1-score, precision, and recall
    df = pd.DataFrame.from_dict(results, orient="index")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Accuracy"], label="Accuracy")
    ax.plot(df["F1-score"], label="F1-score")
    ax.plot(
        df.apply(
            lambda x: x["Confusion Matrix"][1, 1]
            / (x["Confusion Matrix"][1, 1] + x["Confusion Matrix"][0, 1]),
            axis=1,
        ),
        label="Precision",
    )
    ax.plot(
        df.apply(
            lambda x: x["Confusion Matrix"][1, 1]
            / (x["Confusion Matrix"][1, 1] + x["Confusion Matrix"][1, 0]),
            axis=1,
        ),
        label="Recall",
    )
    ax.set_title("Performance Metrics for Different Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.legend()
    # plt.show()

    output_dir = "src/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save the plot in the output directory
    fig.savefig(os.path.join(output_dir, "performance_metrics.png"))

    # show a message to the user indicating the file location
    print(
        f"\nPlot saved at: {os.path.abspath(os.path.join(output_dir, 'performance_metrics.png'))}\n"
    )
