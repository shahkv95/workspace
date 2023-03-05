import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging


def perform_one_hot_encoding(df: pd.DataFrame, categorical: List[str]) -> pd.DataFrame:
    """
    Applying the hot encoding for getting the discrete values
    Perform One Hot Encoding on Categorical Data
    """
    try:
        new_df = pd.get_dummies(df, columns=categorical, drop_first=True)
        logging.info(
            f"One hot encoding performed on categorical columns: {categorical}"
        )
    except Exception as e:
        logging.error(
            f"Error occurred while performing one hot encoding. Error message: {str(e)}"
        )
        raise
    return new_df


def split_data_into_train_test(
    df: pd.DataFrame, test_size: float, target: str
) -> Tuple:
    """
    Splits the dataframe into training and testing datasets
    """
    try:
        X = df.drop([target], axis=1).values
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        logging.info(
            f"Data split into training and testing datasets with test size {test_size}"
        )
    except Exception as e:
        logging.error(
            f"Error occurred while splitting the data into train and test sets. Error message: {str(e)}"
        )
        raise

    return X_train, X_test, y_train, y_test


def get_standardized_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
    """
    Standardizes the data
    """
    try:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        logging.info("Data standardized using StandardScaler")
    except Exception as e:
        logging.error(
            f"Error occurred while standardizing the data. Error message: {str(e)}"
        )
        raise

    return X_train, X_test


def match_test_data_columns(
    ohe_train_columns: pd.Series, test_data: pd.DataFrame, categorical_cols: List[str]
) -> pd.DataFrame:
    """Check if columns of test data after one hot encoding match columns of train data.
    If not, create missing dummy columns in test data and remove extra columns.

    Args:
        train_data: Pandas DataFrame containing training data.
        test_data: Pandas DataFrame containing test data.
        categorical_cols: List of names of categorical columns in the input data.

    Returns:
        Pandas DataFrame with columns matching that of the training data after one hot encoding.
    """
    try:
        test_data_encoded = perform_one_hot_encoding(test_data, categorical_cols)

        if set(ohe_train_columns) != set(test_data_encoded.columns):
            missing_cols = set(ohe_train_columns) - set(test_data_encoded.columns)
            extra_cols = set(test_data_encoded.columns) - set(ohe_train_columns)

            for col in missing_cols:
                test_data_encoded[col] = 0

            test_data_encoded.drop(extra_cols, axis=1, inplace=True)
            test_data_encoded.drop("left", axis=1, inplace=True)

        logging.info(
            "Test data columns matched with training data after one hot encoding"
        )
    except Exception as e:
        logging.error(
            f"Error occurred while matching test data columns with training data. Error message: {str(e)}"
        )
        raise

    return test_data_encoded
