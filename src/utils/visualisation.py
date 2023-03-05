import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation="vertical")
    plt.yticks(range(len(corr.columns)), corr.columns)
    # plt.show()
    save_output_image(VISUALIZATION_OUTPUT_DIR, fig, COR_MATRIX_FILE_NAME)


def plot_heat_map(df: pd.DataFrame) -> None:
    """Plot Heat Map using seaborn.

    Args:
        df (pd.DataFrame): The dataframe to plot the heat map from.

    Returns:
        None
    """
    fig = sns.heatmap(df.corr(), annot=True, fmt=".1g", linewidths=3, linecolor="black")
    # plt.show()
    save_output_image(VISUALIZATION_OUTPUT_DIR, fig.get_figure(), HEATMAP_FILE_NAME)


def plot_pair_plot(df: pd.DataFrame) -> None:
    """Plot Pair Plot using seaborn.

    Args:
        df (pd.DataFrame): The dataframe to plot the pair plot from.

    Returns:
        None
    """
    fig = sns.pairplot(df)
    # plt.show()
    save_output_image(VISUALIZATION_OUTPUT_DIR, fig, PAIR_PLOT_FILE_NAME)
