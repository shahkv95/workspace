import os
import logging
from matplotlib.figure import Figure
import pandas as pd

logging.basicConfig(level=logging.INFO)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        filename = os.path.abspath(file_path)
        df = pd.read_csv(filename)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error("An error occurred while loading data from the file.")
        logging.exception(e)


def save_output_image(output_dir: str, fig: Figure, filename: str) -> None:
    try:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        output_file_path = os.path.join(output_dir, f"{filename}.png")

        # save the plot in the output directory
        fig.savefig(output_file_path)

        # show a message to the user indicating the file location
        logging.info(f"Plot saved at: {os.path.abspath(output_file_path)}")
    except IsADirectoryError as e:
        logging.error(f"Error: {output_dir} is not a directory.")
        logging.exception(e)
    except PermissionError as e:
        logging.error(f"Error: permission denied to write to {output_file_path}.")
        logging.exception(e)
    except Exception as e:
        logging.error("An unknown error occurred while saving the output image.")
        logging.exception(e)
