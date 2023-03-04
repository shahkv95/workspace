import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config.config import FILE_PATH, TARGET, TEST_SIZE
from models.adaboost import fit_ada_boost_model
from models.all_models import fit_all_models
from models.decision_tree import fit_decision_tree_model
from models.k_nearest_neighbors import fit_knn_model
from models.logistic_regression import fit_logistic_regression_model
from models.mlp import fit_mlp_classifier_model
from models.naive_bayes import fit_naive_bayes_model
from models.qda import fit_qda_model
from models.random_forest import fit_random_forest_model
from models.svc import fit_SVC_model
from utils.data_preparation import (
    get_standardized_data,
    perform_one_hot_encoding,
    split_data_into_train_test,
)
from utils.data_preprocessing import (
    get_columns,
    print_columns_null_details,
    print_dataframe_common_details,
    print_dataframe_info,
    print_unique_values_of_columns,
    rename_columns,
    reorder_columns,
)
from utils.utils import load_data
from utils.visualisation import (
    get_correlation_matrix,
    plot_heat_map,
    plot_pair_plot,
)


def main() -> None:
    # Load the data
    df = load_data(FILE_PATH)

    # Data preprocessing
    print_dataframe_common_details(df)

    print_unique_values_of_columns(df, "sales")
    print_unique_values_of_columns(df, "salary")

    columns = get_columns()

    print_columns_null_details(df, columns)

    new_df = rename_columns(df)
    new_df = reorder_columns(new_df)

    # Exploratory Data Analysis
    get_correlation_matrix(new_df)
    plot_heat_map(new_df)
    plot_pair_plot(new_df)

    # Data Preparation for Machine Learning Models
    print_unique_values_of_columns(new_df, "department")
    print_unique_values_of_columns(new_df, "income")

    categorical = ["department", "income"]
    new_df = perform_one_hot_encoding(new_df, categorical)
    print_dataframe_info(new_df)

    X_train, X_test, y_train, y_test = split_data_into_train_test(
        new_df, TEST_SIZE, TARGET
    )

    X_train, X_test = get_standardized_data(X_train, X_test)

    print_dataframe_common_details(df)

    # Model training and testing
    fit_ada_boost_model(X_train, X_test, y_train, y_test)
    fit_decision_tree_model(X_train, X_test, y_train, y_test)
    fit_knn_model(X_train, X_test, y_train, y_test)
    fit_logistic_regression_model(X_train, X_test, y_train, y_test)
    fit_mlp_classifier_model(X_train, X_test, y_train, y_test)
    fit_naive_bayes_model(X_train, X_test, y_train, y_test)
    fit_qda_model(X_train, X_test, y_train, y_test)
    fit_random_forest_model(X_train, X_test, y_train, y_test)
    fit_SVC_model(X_train, X_test, y_train, y_test)
    fit_all_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
