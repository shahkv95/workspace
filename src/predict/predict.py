from typing import List
import pandas as pd
import numpy as np
import os
from config.config import FILE_PATH
from models.model_list import get_list_of_models
from src.utils.data_preprocessing import rename_columns
from utils.data_preparation import match_test_data_columns, perform_one_hot_encoding
from joblib import dump, load

from utils.utils import load_data


def predict_on_test_data(
    ohe_train_columns: pd.Series, test_data_file: str
) -> List[int]:
    # Load test data
    print(test_data_file)
    test_data = load_data(test_data_file)

    print(test_data.columns)

    # Apply preprocessing steps
    test_data = rename_columns(test_data)

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

    categorical = ["department", "income"]
    # test_data = perform_one_hot_encoding(test_data, categorical)

    # train_data = load_data(FILE_PATH)
    test_data = match_test_data_columns(ohe_train_columns, test_data, categorical)

    print("==========    CHECK 0    ==========")
    print(test_data.head())

    MODELS = get_list_of_models()

    # Get user's desired model
    print("Choose a model to predict with: ")
    for i, model_name in enumerate(MODELS.keys()):
        print(f"{i+1}: {model_name}")
    model_choice = int(input("Enter the number of the model you want to use: "))

    print("==========    CHECK 1    ==========")

    selected_model_name = list(MODELS.keys())[model_choice - 1]
    print(selected_model_name)

    print("==========    CHECK 2    ==========")

    selected_model = MODELS[selected_model_name]
    print(selected_model)

    print("==========    CHECK 3    ==========")

    # Check if weights for selected model exist, if not, train the model and save its weights
    model_weights_path = os.path.join(
        "src", "models", "model_weights", f"{selected_model_name}_model_weights.joblib"
    )
    print(model_weights_path)

    print("==========    CHECK 4    ==========")

    if os.path.exists(model_weights_path):
        model = load(model_weights_path)

    print("==========    CHECK 4    ==========")
    print(test_data.head())
    # Predict on test data
    predictions = model.predict(test_data)
    return predictions.tolist()