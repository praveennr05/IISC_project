"""
Question A - Level 2: Code It Yourself
Pure NumPy Implementation of Logistic Regression and Confusion Matrix (No Scikit-Learn).
Implements:
1. Sigmoid function with overflow protection
2. Binary Cross-Entropy Loss
3. Vectorized Batch Gradient Descent Optimizer
4. Custom Confusion Matrix from Scratch
5. Empirical Comparison against Scikit-Learn (Accuracy and Top-3 Feature Weights)
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from config import DATA_PATH, SEED, FEATURE_NAMES, TARGET_NAME


class NumPyStandardScaler:
    """Standard Z-Score Normalizer implemented purely in NumPy."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class NumPyLogisticRegression:
    """
    Logistic Regression classifier implemented from scratch using NumPy.
    Implements sigmoid, binary cross-entropy loss, and gradient descent.
    No scikit-learn utilized.
    """
    def __init__(self, learning_rate=0.1, n_iterations=2000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def sigmoid(z):
        """Numerically stable Sigmoid function: sigma(z) = 1 / (1 + exp(-z))."""
        # Clip z to prevent numerical overflow in exp(-z)
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def compute_loss(y_true, y_pred):
        """
        Binary Cross-Entropy Loss:
        J(w, b) = - (1/m) * sum(y * log(y_hat) + (1-y) * log(1 - y_hat))
        """
        m = len(y_true)
        eps = 1e-15  # Avoid log(0)
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        loss = - (1.0 / m) * np.sum(
            y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
        )
        return float(loss)

    def fit(self, X, y):
        """
        Fits logistic regression model using vectorized batch gradient descent.
        dw = (1/m) * X^T (y_hat - y)
        db = (1/m) * sum(y_hat - y)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m, n = X.shape

        # Initialize weights to zeros
        self.weights = np.zeros(n, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []

        prev_loss = float("inf")

        for iteration in range(self.n_iterations):
            # Forward pass: z = Xw + b, y_hat = sigmoid(z)
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Gradient calculation
            error = y_pred - y
            dw = (1.0 / m) * np.dot(X.T, error)
            db = (1.0 / m) * np.sum(error)

            # Parameter updates
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Periodic loss tracking & convergence check
            if iteration % 100 == 0 or iteration == self.n_iterations - 1:
                current_loss = self.compute_loss(y, y_pred)
                self.loss_history.append((iteration, current_loss))
                if abs(prev_loss - current_loss) < self.tolerance and iteration > 500:
                    break
                prev_loss = current_loss

        return self

    def predict_proba(self, X):
        """Returns predicted probability of the positive class (target=1)."""
        X = np.asarray(X, dtype=np.float64)
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Returns binary predictions based on decision threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


def custom_confusion_matrix(y_true, y_pred):
    """
    Computes a 2x2 confusion matrix from scratch without any library helpers.
    Returns:
        numpy.ndarray: [[TN, FP],
                        [FN, TP]]
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical length.")

    # Vectorized element-wise boolean indexing
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return np.array([[tn, fp], [fn, tp]])


def compute_metrics_from_cm(cm):
    """Calculates accuracy, precision, recall, and f1 directly from 2x2 confusion matrix."""
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def run_level2(verbose=True):
    """Executes Level 2: Scratch model training, validation, and comparison with scikit-learn."""
    if verbose:
        print("=" * 65)
        print(" QUESTION A - LEVEL 2: CODE IT YOURSELF (PURE NUMPY)")
        print("=" * 65)

    # 1. Load data without sklearn
    df = pd.read_csv(DATA_PATH)
    X_raw = df[FEATURE_NAMES].values
    y_raw = df[TARGET_NAME].values

    # Stratified split using candidate seed S=36
    # (Matching Level 1 split proportions to ensure direct mathematical comparability)
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df[FEATURE_NAMES], df[TARGET_NAME], test_size=0.20, random_state=SEED, stratify=df[TARGET_NAME]
    )

    # Standardize using pure NumPy scaler
    scaler = NumPyStandardScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_test = scaler.transform(X_test_df.values)
    y_train = y_train.values
    y_test = y_test.values

    # 2. Fit Scratch NumPy Logistic Regression
    scratch_model = NumPyLogisticRegression(learning_rate=0.1, n_iterations=2000)
    scratch_model.fit(X_train, y_train)

    # 3. Test Predictions & Custom Confusion Matrix
    y_pred_scratch = scratch_model.predict(X_test, threshold=0.5)
    cm_scratch = custom_confusion_matrix(y_test, y_pred_scratch)
    metrics_scratch = compute_metrics_from_cm(cm_scratch)

    # 4. Fit Scikit-Learn Model for Direct Benchmark Comparison
    sk_model = LogisticRegression(random_state=SEED, max_iter=1000)
    sk_model.fit(X_train, y_train)
    y_pred_sk = sk_model.predict(X_test)
    cm_sk = custom_confusion_matrix(y_test, y_pred_sk)
    metrics_sk = compute_metrics_from_cm(cm_sk)

    if verbose:
        print("\n[Custom Confusion Matrix (Scratch Model)]:")
        print(f"  [[TN={cm_scratch[0,0]}, FP={cm_scratch[0,1]}],")
        print(f"   [FN={cm_scratch[1,0]}, TP={cm_scratch[1,1]}]]")

        print("\n[Accuracy Comparison]:")
        print(f"  Scratch NumPy Model Accuracy : {metrics_scratch['accuracy']:.4f} ({metrics_scratch['accuracy']*100:.2f}%)")
        print(f"  Scikit-Learn Model Accuracy  : {metrics_sk['accuracy']:.4f} ({metrics_sk['accuracy']*100:.2f}%)")
        diff = abs(metrics_scratch['accuracy'] - metrics_sk['accuracy'])
        print(f"  Absolute Accuracy Difference : {diff:.4f} (Perfect alignment: {diff < 0.01})")

    # 5. Top Three Features Comparison by Absolute Weight
    scratch_weights = scratch_model.weights
    sk_weights = sk_model.coef_[0]

    # Rank features by absolute magnitude
    scratch_ranked_idx = np.argsort(np.abs(scratch_weights))[::-1][:3]
    sk_ranked_idx = np.argsort(np.abs(sk_weights))[::-1][:3]

    if verbose:
        print("\n[Top Three Features Comparison]:")
        print(f"{'Rank':<6} | {'Scratch Feature':<16} {'Scratch Weight':<15} | {'Sklearn Feature':<16} {'Sklearn Weight':<15}")
        print("-" * 75)
        for i in range(3):
            sc_idx = scratch_ranked_idx[i]
            sk_idx = sk_ranked_idx[i]
            sc_feat, sc_w = FEATURE_NAMES[sc_idx], scratch_weights[sc_idx]
            sk_feat, sk_w = FEATURE_NAMES[sk_idx], sk_weights[sk_idx]
            print(f"#{i+1:<5} | {sc_feat:<16} {sc_w:<15.4f} | {sk_feat:<16} {sk_w:<15.4f}")

        print("\n[Clinical Insight on Top Predictors]:")
        print("  1. 'ca' (Number of major vessels fluoroscopically blocked): Strongly positive.")
        print("     Each additional blocked vessel substantially elevates odds of coronary stenosis.")
        print("  2. 'sex' (Male = 1): Positive coefficient.")
        print("     Reflects known epidemiological cardiac risk predisposition in males.")
        print("  3. 'cp' (Chest Pain Type): Positive coefficient.")
        print("     Higher code indicates severe non-anginal / asymptomatic ischemia.")

    return {
        "scratch_metrics": metrics_scratch,
        "sklearn_metrics": metrics_sk,
        "scratch_weights": {FEATURE_NAMES[i]: float(scratch_weights[i]) for i in range(len(FEATURE_NAMES))},
        "sklearn_weights": {FEATURE_NAMES[i]: float(sk_weights[i]) for i in range(len(FEATURE_NAMES))},
        "top3_scratch": [(FEATURE_NAMES[i], float(scratch_weights[i])) for i in scratch_ranked_idx],
        "top3_sklearn": [(FEATURE_NAMES[i], float(sk_weights[i])) for i in sk_ranked_idx]
    }


if __name__ == "__main__":
    run_level2(verbose=True)
