import os
from typing import List

import pandas as pd

from utils.utils import load_data


def print_dataframe_common_details(df: pd.DataFrame) -> None:
    """Print dataframe details such as head, tail, info, etc."""

    print("\n========================  SHAPE  ========================\n")
    print("DataFrame Shape: ", df.shape)

    print("\n========================  HEAD  ========================\n")
    print(df.head())

    print("\n========================  TAIL  ========================\n")
    print(df.tail())

    print("\n========================  INFO  ========================\n")
    print(df.info())

    print("\n========================  STATISTICS  ========================\n")
    print(df.describe())


def print_dataframe_info(df: pd.DataFrame) -> None:
    print("\n========================  INFO  ========================\n")
    print(df.info())


def print_unique_values_of_columns(df: pd.DataFrame, column: str) -> None:
    print("\n========================  UNIQUE_VALUES  ========================\n")
    print("Unique Values for Column: ", column)
    print(df[column].unique())
    print("\nTotal number of unique values: ", len(df[column].unique()))


def get_columns() -> List[str]:
    columns = [
        "last_evaluation",
        "number_project",
        "promotion_last_5years",
        "sales",
        "salary",
        "time_spend_company",
        "Work_accident",
        "satisfaction_level",
        "left",
    ]
    return columns


def print_columns_null_details(df: pd.DataFrame, columns: List[str]) -> None:
    """Print null details of selected columns."""
    print(
        "\n========================  COUNT_OF_NULL_VALUES  ========================\n"
    )
    print(df[columns].isnull())
    print(df[columns].isnull().sum())


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns of the dataframe."""
    new_df = df.rename(
        columns={
            "sales": "department",
            "salary": "income",
            "Work_accident": "work_accident",
        }
    )
    return new_df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns of the dataframe."""
    new_df = df[
        [
            "department",
            "time_spend_company",
            "average_montly_hours",
            "number_project",
            "last_evaluation",
            "promotion_last_5years",
            "work_accident",
            "income",
            "satisfaction_level",
            "left",
        ]
    ]
    return new_df
