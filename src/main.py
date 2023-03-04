import os
import pandas as pd

# Loading file in the main.py using absolute file path
filename = os.path.abspath("src/data/employee_retention.csv")
df = pd.read_csv(filename)

print(df.head())
