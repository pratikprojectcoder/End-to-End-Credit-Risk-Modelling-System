"""Train and persist credit risk models and preprocessing artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from xgboost import XGBClassifier


# =========================================================
# PROJECT IMPORTS
# =========================================================

_SRC = Path(__file__).resolve().parent

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preprocess import MODELS_DIR, PROCESSED_DATA_DIR
from utils import save_pickle


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(y_true, y_pred, model_name):
    """
    Print evaluation metrics for a model.
    """

    print(f"\n{'=' * 60}")
    print(f"{model_name}")
    print(f"{'=' * 60}")

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("ROC AUC  :", roc_auc_score(y_true, y_pred))

    print("\nClassification Report:\n")

    print(classification_report(y_true, y_pred))


# =========================================================
# TRAINING PIPELINE
# =========================================================

def train(
    processed_path: Path | None = None,
    target_column: str = "target",
) -> None:
    """
    Load processed dataset, train models,
    evaluate performance, and save artifacts.
    """

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    processed_path = processed_path or (
        PROCESSED_DATA_DIR / "train.parquet"
    )

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {processed_path}"
        )

    df = pd.read_parquet(processed_path)

    # -----------------------------------------------------
    # VALIDATE TARGET COLUMN
    # -----------------------------------------------------

    if target_column not in df.columns:
        raise ValueError(
            f"Missing target column {target_column!r}"
        )

    # -----------------------------------------------------
    # FEATURES AND TARGET
    # -----------------------------------------------------

    feature_cols = [
        c for c in df.columns
        if c != target_column
    ]

    X = (
        df[feature_cols]
        .select_dtypes(include=[np.number])
        .fillna(0)
    )

    y = df[target_column]

    print("\nDataset Loaded Successfully.")
    print(f"Dataset Shape: {df.shape}")

    # -----------------------------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain-Test Split Completed.")
    print(f"X_train Shape: {X_train.shape}")
    print(f"X_test Shape : {X_test.shape}")

    # -----------------------------------------------------
    # FEATURE SCALING
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature Scaling Applied.")

    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

    print("\nTraining Logistic Regression...")

    lr_model = LogisticRegression(
        max_iter=1000
    )

    lr_model.fit(
        X_train_scaled,
        y_train
    )

    lr_preds = lr_model.predict(
        X_test_scaled
    )

    evaluate_model(
        y_test,
        lr_preds,
        "Logistic Regression"
    )

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    print("\nTraining Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    rf_model.fit(
        X_train,
        y_train
    )

    rf_preds = rf_model.predict(
        X_test
    )

    evaluate_model(
        y_test,
        rf_preds,
        "Random Forest"
    )

    # =====================================================
    # XGBOOST
    # =====================================================

    print("\nTraining XGBoost...")

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )

    xgb_model.fit(
        X_train,
        y_train
    )

    xgb_preds = xgb_model.predict(
        X_test
    )

    evaluate_model(
        y_test,
        xgb_preds,
        "XGBoost"
    )

    # =====================================================
    # SAVE ARTIFACTS
    # =====================================================

    encoders_payload = {
        "label_encoders": {},
        "feature_columns": list(X.columns),
    }

    save_pickle(
        xgb_model,
        MODELS_DIR / "xgboost_model.pkl"
    )

    save_pickle(
        scaler,
        MODELS_DIR / "scaler.pkl"
    )

    save_pickle(
        encoders_payload,
        MODELS_DIR / "label_encoders.pkl"
    )

    print("\nModels and preprocessing artifacts saved successfully.")

    print("\nSaved Files:")
    print("- xgboost_model.pkl")
    print("- scaler.pkl")
    print("- label_encoders.pkl")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train()