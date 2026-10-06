"""
Question A - Level 1: Build
Predicting Health Risk using UCI Heart Disease dataset.
Trains Scikit-Learn Logistic Regression and Random Forest models with Candidate Seed S=36.
Reports Accuracy, Precision, and Recall for both models.
Exports trained model and scaler for Question B deployment.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import DATA_PATH, SEED, FEATURE_NAMES, TARGET_NAME, MODEL_PATH, SCALER_PATH


def load_and_preprocess_data():
    """Loads cleaned heart dataset and splits into train and test sets with seed S."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run data/prepare_data.py first.")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    # Stratified 80/20 train/test split using candidate seed S=36
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # Standardize numerical features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def run_level1(save_artifacts=True):
    """Executes Level 1: trains LR & RF, reports metrics, and exports artifacts."""
    print("=" * 65)
    print(f" QUESTION A - LEVEL 1: BUILD (Random Seed S = {SEED})")
    print("=" * 65)

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data()
    print(f"Total dataset: {len(X_train) + len(X_test)} samples")
    print(f"Training set : {len(X_train)} samples")
    print(f"Test set     : {len(X_test)} samples ({y_test.sum()} positive, {len(y_test) - y_test.sum()} negative)")

    # 1. Train Scikit-Learn Logistic Regression
    lr = LogisticRegression(random_state=SEED, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    lr_acc = float(accuracy_score(y_test, y_pred_lr))
    lr_prec = float(precision_score(y_test, y_pred_lr, zero_division=0))
    lr_rec = float(recall_score(y_test, y_pred_lr, zero_division=0))
    lr_f1 = float(f1_score(y_test, y_pred_lr, zero_division=0))
    lr_cm = confusion_matrix(y_test, y_pred_lr).tolist()

    print("\n[Model 1: Scikit-Learn Logistic Regression]")
    print(f"  Accuracy : {lr_acc:.4f} ({lr_acc * 100:.2f}%)")
    print(f"  Precision: {lr_prec:.4f} ({lr_prec * 100:.2f}%)")
    print(f"  Recall   : {lr_rec:.4f} ({lr_rec * 100:.2f}%)")
    print(f"  F1 Score : {lr_f1:.4f}")
    print(f"  Confusion Matrix (TN, FP / FN, TP):\n    {lr_cm[0]}\n    {lr_cm[1]}")

    # 2. Train Scikit-Learn Random Forest
    rf = RandomForestClassifier(random_state=SEED, n_estimators=100, max_depth=5)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rf_acc = float(accuracy_score(y_test, y_pred_rf))
    rf_prec = float(precision_score(y_test, y_pred_rf, zero_division=0))
    rf_rec = float(recall_score(y_test, y_pred_rf, zero_division=0))
    rf_f1 = float(f1_score(y_test, y_pred_rf, zero_division=0))
    rf_cm = confusion_matrix(y_test, y_pred_rf).tolist()

    print("\n[Model 2: Scikit-Learn Random Forest]")
    print(f"  Accuracy : {rf_acc:.4f} ({rf_acc * 100:.2f}%)")
    print(f"  Precision: {rf_prec:.4f} ({rf_prec * 100:.2f}%)")
    print(f"  Recall   : {rf_rec:.4f} ({rf_rec * 100:.2f}%)")
    print(f"  F1 Score : {rf_f1:.4f}")
    print(f"  Confusion Matrix (TN, FP / FN, TP):\n    {rf_cm[0]}\n    {rf_cm[1]}")

    # Export artifacts for Question B if requested
    if save_artifacts:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(lr, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"\nSaved production model to: {MODEL_PATH}")
        print(f"Saved production scaler to: {SCALER_PATH}")

    results = {
        "seed": SEED,
        "logistic_regression": {
            "accuracy": lr_acc,
            "precision": lr_prec,
            "recall": lr_rec,
            "f1": lr_f1,
            "confusion_matrix": lr_cm
        },
        "random_forest": {
            "accuracy": rf_acc,
            "precision": rf_prec,
            "recall": rf_rec,
            "f1": rf_f1,
            "confusion_matrix": rf_cm
        }
    }
    return results


if __name__ == "__main__":
    run_level1(save_artifacts=True)
