# Displaying the correlation matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_correlation_matrix(df, size=10):
    """Returns the correlation matrix of the dataframe"""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation="vertical")
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.show()


def plot_heat_map(df):
    """Plot Heat Map using seaborn"""
    sns.heatmap(df.corr(), annot=True, fmt=".1g", linewidths=3, linecolor="black")
    plt.show()


def plot_pair_plot(df):
    """Plot Pair Plot using seaborn"""
    sns.pairplot(df)
    plt.show()
