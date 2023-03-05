import logging
from typing import List
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from config.config import FILE_PATH
from models.model_list import get_list_of_models
from src.utils.data_preprocessing import rename_columns
from utils.data_preparation import match_test_data_columns, perform_one_hot_encoding
from joblib import dump, load

from utils.utils import load_data

logging.basicConfig(
    filename="app.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def predict_on_test_data(ohe_train_columns: pd.Series, test_data_file: str) -> None:
    try:
        # Load test data
        logging.info("Loading test data......\n")
        test_data = load_data(test_data_file)

        # Apply preprocessing steps
        logging.info("Preparing test data......\n")
        logging.info("Renaming expected column names\n")
        test_data = rename_columns(test_data)

        logging.info("Reordering column names in the expected order......\n")
        test_data = test_data[
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
            ]
        ]

        logging.info("Appying one hot encoding and post data processing on test data\n")
        categorical = ["department", "income"]
        test_data = match_test_data_columns(ohe_train_columns, test_data, categorical)

        sc = StandardScaler()
        test_data = sc.fit_transform(test_data)

        MODELS = get_list_of_models()

        while True:
            try:
                # Get user's desired model
                logging.info(
                    "Asking user to provide a model choice for predicting the test data\n"
                )
                print("Choose a model to predict with: ")
                for i, model_name in enumerate(MODELS.keys()):
                    print(f"{i+1}: {model_name}")
                model_choice = int(
                    input("Enter the number of the model you want to use: ")
                )

                selected_model_name = list(MODELS.keys())[model_choice - 1]

                model_weights_path = os.path.join(
                    "src",
                    "models",
                    "model_weights",
                    f"{selected_model_name}_model_weights.joblib",
                )

                logging.info(
                    "Checking if the model weights already exists in the path \n\n"
                )
                if os.path.exists(model_weights_path):
                    model = load(model_weights_path)
                else:
                    logging.error(
                        f"Model weights not found for model: {selected_model_name}"
                    )
                    continue

                # Predict on test data
                predictions = model.predict(test_data)
                logging.info(
                    f"Prediction using {selected_model_name}: {predictions.tolist()}"
                )

                user_choice = input("Do you want to predict again? (y/n) ")
                if user_choice.lower() == "n":
                    break
            except ValueError as ve:
                logging.error(f"Invalid input entered by user: {ve}", exc_info=True)
                print("Invalid input entered. Please try again.")
            except IndexError as ie:
                logging.error(
                    f"Invalid model choice entered by user: {ie}", exc_info=True
                )
                print("Invalid model choice entered. Please try again.")
            except Exception as e:
                logging.error(f"Error occurred: {e}", exc_info=True)
                print("An error occurred. Please try again.")
    except FileNotFoundError as fnf:
        logging.error(f"File not found: {fnf}", exc_info=True)
        print("File not found. Please check file path and try again.")
    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        print("An error occurred. Please try again.")
