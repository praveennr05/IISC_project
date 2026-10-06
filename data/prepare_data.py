"""
Data Preparation and Cleaning Script
Dataset: UCI Heart Disease (Cleveland Clinic Foundation)
Provenance: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
"""

import os
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(DATA_DIR, "raw_download.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "heart.csv")

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age",       # Age in years (29 - 77)
    "sex",       # 1 = male, 0 = female
    "cp",        # Chest pain type: 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic
    "trestbps",  # Resting blood pressure in mm Hg (94 - 200)
    "chol",      # Serum cholesterol in mg/dl (126 - 564)
    "fbs",       # Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",   # Resting ECG: 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy
    "thalach",   # Maximum heart rate achieved (71 - 202)
    "exang",     # Exercise induced angina (1 = yes, 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest (0.0 - 6.2)
    "slope",     # Slope of peak exercise ST segment: 1 = upsloping, 2 = flat, 3 = downsloping
    "ca",        # Number of major vessels (0-3) colored by fluoroscopy
    "thal",      # Thalassemia: 3 = normal, 6 = fixed defect, 7 = reversible defect
    "target"     # Diagnosis: 0 = <50% narrowing (healthy), 1-4 = >50% narrowing (disease present)
]


def download_and_clean():
    if not os.path.exists(OUTPUT_PATH):
        print(f"Downloading raw data from {UCI_URL}...")
        req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
            with open(RAW_PATH, "w", encoding="utf-8") as f:
                f.write(content)
        print("Download complete.")
    else:
        print(f"Target dataset {OUTPUT_PATH} already exists.")

    if os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_PATH, header=None, names=COLUMNS, na_values="?")
    else:
        df = pd.read_csv(OUTPUT_PATH)

    print(f"Initial raw shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Mode imputation for missing values in ca (4 records) and thal (2 records)
    if "ca" in df and df["ca"].isnull().sum() > 0:
        df["ca"] = df["ca"].fillna(df["ca"].mode()[0])
    if "thal" in df and df["thal"].isnull().sum() > 0:
        df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

    # Binarize target: 0 = healthy (no heart disease), 1 = disease present
    df["target"] = (df["target"] > 0).astype(int)

    # Cast integer and float types cleanly
    int_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "slope", "ca", "thal", "target"]
    for col in int_cols:
        df[col] = df[col].astype(int)
    df["oldpeak"] = df["oldpeak"].astype(float)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully processed and saved {len(df)} records to {OUTPUT_PATH}")

    # Clean up raw download if present
    if os.path.exists(RAW_PATH):
        os.remove(RAW_PATH)


if __name__ == "__main__":
    download_and_clean()
