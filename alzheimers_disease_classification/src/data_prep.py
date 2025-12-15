import pandas as pd
import os

def load_and_clean_data():
    # Get absolute path of this file (data_prep.py)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Go to project root (one level up from src)
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    # Build full path to data file
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "alzheimers_disease_data.csv")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Drop useless columns
    df.drop(columns=["PatientID", "DoctorInCharge"], inplace=True)

    # Separate features and target
    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    return X, y


if __name__ == "__main__":
    X, y = load_and_clean_data()

    print("Data loaded successfully")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
