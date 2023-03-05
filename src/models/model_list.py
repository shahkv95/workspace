from typing import Dict, Type
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from models.decision_tree import DecisionTreeClassifier
from models.random_forest import RandomForestClassifier


def get_list_of_models() -> Dict[str, Type]:
    MODELS: Dict[str, Type] = {
        "adaboost": AdaBoostClassifier,
        "decision_tree": DecisionTreeClassifier,
        "k_nearest_neighbor": KNeighborsClassifier,
        "logistic_regression": LogisticRegression,
        "mlp": MLPClassifier,
        "naive_bayes": GaussianNB,
        "qda": QuadraticDiscriminantAnalysis,
        "random_forest": RandomForestClassifier,
        "svc": SVC,
    }

    return MODELS
