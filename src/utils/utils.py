import os
from matplotlib.figure import Figure
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    filename = os.path.abspath(file_path)
    df = pd.read_csv(filename)
    return df


def save_output_image(output_dir: str, fig: Figure, filename: str) -> None:
    output_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(output_dir, f"{filename}.png")

    # save the plot in the output directory
    fig.savefig(output_file_path)

    # show a message to the user indicating the file location
    print(f"\nPlot saved at: {os.path.abspath(output_file_path)}\n")
