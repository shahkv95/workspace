import logging
from typing import List
import pandas as pd


def print_dataframe_common_details(df: pd.DataFrame) -> None:
    """Print dataframe details such as head, tail, info, etc."""
    logging.info("\n========================  SHAPE  ========================\n")
    logging.info(f"DataFrame Shape: {df.shape}")

    logging.info("\n========================  HEAD  ========================\n")
    logging.info(df.head())

    logging.info("\n========================  TAIL  ========================\n")
    logging.info(df.tail())

    logging.info("\n========================  INFO  ========================\n")
    logging.info(df.info())

    logging.info("\n========================  STATISTICS  ========================\n")
    logging.info(df.describe())


def print_dataframe_info(df: pd.DataFrame) -> None:
    """Print info of dataframe"""
    logging.info("\n========================  INFO  ========================\n")
    logging.info(df.info())


def print_unique_values_of_columns(df: pd.DataFrame, column: str) -> None:
    """Print unique values of a column."""
    logging.info(
        "\n========================  UNIQUE_VALUES  ========================\n"
    )
    logging.info(f"Unique Values for Column: {column}")
    logging.info(df[column].unique())
    logging.info(f"\nTotal number of unique values: {len(df[column].unique())}")


def get_columns() -> List[str]:
    """Return the columns of the dataframe"""
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
    logging.info(
        "\n========================  COUNT_OF_NULL_VALUES  ========================\n"
    )
    logging.info(df[columns].isnull())
    logging.info(df[columns].isnull().sum())


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns of the dataframe."""
    try:
        new_df = df.rename(
            columns={
                "sales": "department",
                "salary": "income",
                "Work_accident": "work_accident",
            }
        )
        return new_df
    except Exception as e:
        logging.error(f"Error renaming columns: {e}")


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns of the dataframe."""
    try:
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
    except Exception as e:
        logging.error(f"Error reordering columns: {e}")
