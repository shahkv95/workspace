import os
import logging
from typing import Any
import joblib
from matplotlib.figure import Figure
import pandas as pd

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data.
    """

    try:
        filename = os.path.abspath(file_path)
        df = pd.read_csv(filename)
        logger.info(f"Data loaded successfully from {filename}")
        return df
    except Exception as e:
        logger.error("An error occurred while loading data from the file.")
        logger.exception(e)


def save_output_image(output_dir: str, fig: Figure, filename: str) -> None:
    """Save the output image to the specified directory.

    Args:
        output_dir (str): The output directory to save the image.
        fig (Figure): The figure object to save.
        filename (str): The filename of the output image.

    Returns:
        None
    """
    logger.info("Saving output image...")

    try:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        output_file_path = os.path.join(output_dir, f"{filename}.png")

        # save the plot in the output directory
        fig.savefig(output_file_path)

        # show a message to the user indicating the file location
        logger.info(f"Plot saved at: {os.path.abspath(output_file_path)}")
    except IsADirectoryError as e:
        logger.error(f"Error: {output_dir} is not a directory.")
        logger.exception(e)
    except PermissionError as e:
        logger.error(f"Error: permission denied to write to {output_file_path}.")
        logger.exception(e)
    except Exception as e:
        logger.error("An unknown error occurred while saving the output image.")
        logger.exception(e)


def save_model_weights(model: Any, model_name: str, __file__: str) -> None:
    """Save model weights using joblib dump function.

    Args:
        model (Any): The model whose weights are to be saved.
        model_name (str): Name of the model.

    Returns:
        None
    """

    logger.info("Saving model weights...")

    try:
        model_weights_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "model_weights"
        )
        os.makedirs(model_weights_dir, exist_ok=True)
        weights_file_path = os.path.join(
            model_weights_dir, f"{model_name}_weights.joblib"
        )
        joblib.dump(model, weights_file_path)
        logger.info(f"Model weights saved at: {os.path.abspath(weights_file_path)}")
    except Exception as e:
        logger.error("An error occurred while saving model weights.")
        logger.exception(e)
