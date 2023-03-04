import os
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    filename = os.path.abspath(file_path)
    df = pd.read_csv(filename)
    return df
