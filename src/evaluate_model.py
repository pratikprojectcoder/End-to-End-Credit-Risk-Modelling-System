"""Evaluate trained credit risk model performance."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =========================================================
# PROJECT IMPORTS
# =========================================================

_SRC = Path(__file__).resolve().parent

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preprocess import MODELS_DIR, PROCESSED_DATA_DIR
from utils import load_pickle


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate(
    processed_path: Path | None = None,
    target_column: str = "loan_status",
) -> dict[str, float]:
    """
    Evaluate trained model on processed dataset.
    """

    # -----------------------------------------------------
    # LOAD DATASET
    # -----------------------------------------------------

    processed_path = processed_path or (
        PROCESSED_DATA_DIR /
        "balanced_train_data.csv"
    )

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {processed_path}"
        )

    df = pd.read_csv(processed_path)

    print("\nDataset Loaded Successfully.")
    print(f"Dataset Shape: {df.shape}")

    # -----------------------------------------------------
    # LOAD SAVED ARTIFACTS
    # -----------------------------------------------------

    model = load_pickle(
        MODELS_DIR / "xgboost_model.pkl"
    )

    scaler = load_pickle(
        MODELS_DIR / "scaler.pkl"
    )

    encoders = load_pickle(
        MODELS_DIR / "label_encoders.pkl"
    )

    if model is None:
        raise RuntimeError(
            "Model is not trained yet."
        )

    # -----------------------------------------------------
    # FEATURES AND TARGET
    # -----------------------------------------------------

    feature_cols = encoders.get(
        "feature_columns",
        []
    )

    X = (
        df.reindex(
            columns=feature_cols,
            fill_value=0
        )
        .select_dtypes(include=[np.number])
        .fillna(0)
    )

    y = df[target_column]

    print("\nFeatures and target prepared.")

    # -----------------------------------------------------
    # SCALE FEATURES
    # -----------------------------------------------------

    X_scaled = scaler.transform(X)

    print("Feature scaling applied.")

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    proba = model.predict_proba(X_scaled)[:, 1]

    pred = (
        proba >= 0.5
    ).astype(int)

    print("Predictions generated.")

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    metrics = {

        "accuracy": accuracy_score(
            y,
            pred
        ),

        "precision": precision_score(
            y,
            pred
        ),

        "recall": recall_score(
            y,
            pred
        ),

        "f1_score": f1_score(
            y,
            pred
        ),

        "roc_auc": roc_auc_score(
            y,
            proba
        ),
    }

    # -----------------------------------------------------
    # PRINT METRICS
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL EVALUATION METRICS")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key:<12}: {value:.4f}")

    # -----------------------------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y,
            pred,
            digits=4
        )
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    cm = confusion_matrix(
        y,
        pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.show()

    return metrics


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    evaluate()