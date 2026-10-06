"""
Global Configuration for Technical Assignment: AI for Personal Health and Wellness
Candidate Personal Seed: S = 36 (Derived from last digits of USN: 036)
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "heart.csv"

# Models and Artifacts
QUESTION_A_DIR = BASE_DIR / "question_a"
QUESTION_B_DIR = BASE_DIR / "question_b"
MODELS_DIR = QUESTION_B_DIR / "models"
MODEL_PATH = MODELS_DIR / "heart_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_PATH = QUESTION_A_DIR / "results_summary.json"

# SQLite Database for Question B
DB_PATH = QUESTION_B_DIR / "health_records.db"

# Mandatory Candidate Seed
SEED = 36

# Feature Specifications (UCI Cleveland Heart Disease)
FEATURE_NAMES = [
    "age",       # Age in years (1 - 120)
    "sex",       # Biological sex (0 = Female, 1 = Male)
    "cp",        # Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal, 4 = asymptomatic)
    "trestbps",  # Resting blood pressure in mm Hg (50 - 300)
    "chol",      # Serum cholesterol in mg/dl (100 - 600)
    "fbs",       # Fasting blood sugar > 120 mg/dl (0 = False, 1 = True)
    "restecg",   # Resting ECG (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)
    "thalach",   # Maximum heart rate achieved (50 - 250)
    "exang",     # Exercise induced angina (0 = No, 1 = Yes)
    "oldpeak",   # ST depression induced by exercise relative to rest (0.0 - 10.0)
    "slope",     # Slope of peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)
    "ca",        # Major vessels (0 - 3) colored by fluoroscopy
    "thal"       # Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
]

TARGET_NAME = "target"
