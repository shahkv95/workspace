import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging

from config.config import (
    COR_MATRIX_FILE_NAME,
    HEATMAP_FILE_NAME,
    PAIR_PLOT_FILE_NAME,
    VISUALIZATION_OUTPUT_DIR,
)
from utils.utils import save_output_image


def get_correlation_matrix(df: pd.DataFrame, size: int = 10) -> None:
    """Returns the correlation matrix of the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to get the correlation matrix from.
        size (int, optional): The size of the plot. Defaults to 10.

    Returns:
        None
    """
    try:
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(size, size))
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation="vertical")
        plt.yticks(range(len(corr.columns)), corr.columns)
        save_output_image(VISUALIZATION_OUTPUT_DIR, fig, COR_MATRIX_FILE_NAME)
        logging.info(
            f"Correlation matrix plot saved at: {VISUALIZATION_OUTPUT_DIR}/{COR_MATRIX_FILE_NAME}.png"
        )
    except Exception as e:
        logging.error(
            f"An error occurred while generating the correlation matrix plot: {e}"
        )


def plot_heat_map(df: pd.DataFrame) -> None:
    """Plot Heat Map using seaborn.

    Args:
        df (pd.DataFrame): The dataframe to plot the heat map from.

    Returns:
        None
    """
    try:
        fig = sns.heatmap(
            df.corr(), annot=True, fmt=".1g", linewidths=3, linecolor="black"
        )
        save_output_image(VISUALIZATION_OUTPUT_DIR, fig.get_figure(), HEATMAP_FILE_NAME)
        logging.info(
            f"Heatmap plot saved at: {VISUALIZATION_OUTPUT_DIR}/{HEATMAP_FILE_NAME}.png"
        )
    except Exception as e:
        logging.error(f"An error occurred while generating the heatmap plot: {e}")


def plot_pair_plot(df: pd.DataFrame) -> None:
    """Plot Pair Plot using seaborn.

    Args:
        df (pd.DataFrame): The dataframe to plot the pair plot from.

    Returns:
        None
    """
    try:
        fig = sns.pairplot(df)
        save_output_image(VISUALIZATION_OUTPUT_DIR, fig, PAIR_PLOT_FILE_NAME)
        logging.info(
            f"Pair plot saved at: {VISUALIZATION_OUTPUT_DIR}/{PAIR_PLOT_FILE_NAME}.png"
        )
    except Exception as e:
        logging.error(f"An error occurred while generating the pair plot: {e}")
