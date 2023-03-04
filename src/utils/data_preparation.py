import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def perform_one_hot_encoding(df, categorical):
    """Applying the hot encoding for getting the discrete values"""
    """Perform One Hot Encoding on Categorical Data"""
    new_df = pd.get_dummies(df, columns=categorical, drop_first=True)
    return new_df


def split_data_into_train_test(df, test_size, target):
    """Splits the dataframe into training and testing datasets"""
    X = df.drop([target], axis=1).values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


def get_standardize_data(X_train, X_test):
    """Standardizes the data"""
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test
