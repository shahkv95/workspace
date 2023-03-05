import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config.config import FILE_PATH, TARGET, TEST_DATA_FILE, TEST_SIZE
from models.adaboost import fit_adaboost_model
from models.all_models import fit_all_models
from models.decision_tree import fit_decision_tree_model
from models.k_nearest_neighbors import fit_knn_model
from models.logistic_regression import fit_logistic_regression_model
from models.mlp import fit_mlp_classifier_model
from models.naive_bayes import fit_naive_bayes_model
from models.qda import fit_qda_model
from models.random_forest import fit_random_forest_model
from models.svc import fit_svc_model
from predict.predict import predict_on_test_data
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
    np.random.seed(42)  # set random seed

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
    # get_correlation_matrix(new_df)
    # plot_heat_map(new_df)
    # plot_pair_plot(new_df)

    # Data Preparation for Machine Learning Models
    print_unique_values_of_columns(new_df, "department")
    print_unique_values_of_columns(new_df, "income")

    categorical = ["department", "income"]
    new_df = perform_one_hot_encoding(new_df, categorical)
    ohe_train_columns = new_df.columns
    print_dataframe_info(new_df)

    X_train, X_test, y_train, y_test = split_data_into_train_test(
        new_df, TEST_SIZE, TARGET
    )

    X_train, X_test = get_standardized_data(X_train, X_test)

    print_dataframe_common_details(df)

    # Model training and testing on training data
    fit_adaboost_model(X_train, X_test, y_train, y_test, "adaboost_model")
    fit_decision_tree_model(X_train, X_test, y_train, y_test, "decision_tree_model")
    fit_knn_model(X_train, X_test, y_train, y_test, "k_nearest_neighbors_model")
    fit_logistic_regression_model(
        X_train, X_test, y_train, y_test, "logistic_regression_model"
    )
    fit_mlp_classifier_model(X_train, X_test, y_train, y_test, "mlp_classifier_model")
    fit_naive_bayes_model(X_train, X_test, y_train, y_test, "naive_bayes_model")
    fit_qda_model(X_train, X_test, y_train, y_test, "qda_model")
    fit_random_forest_model(X_train, X_test, y_train, y_test, "random_forest_model")
    fit_svc_model(X_train, X_test, y_train, y_test, "svc_model")
    fit_all_models(X_train, X_test, y_train, y_test)

    # Model testing on testing dataset - predictions
    test_data_file = TEST_DATA_FILE
    predict_on_test_data(ohe_train_columns, test_data_file)


if __name__ == "__main__":
    main()
