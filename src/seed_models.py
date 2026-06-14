"""Seed demo model artifacts for immediate API and dashboard use."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preprocess import MODELS_DIR
from utils import save_pickle

FEATURE_COLUMNS = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "fico_range_low",
    "revol_bal",
    "installment",
    "delinq_2yrs",
    "pub_rec",
]


def _generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    loan_amnt = rng.integers(1000, 40000, n_samples).astype(float)
    annual_inc = rng.integers(20000, 150000, n_samples).astype(float)
    dti = rng.uniform(5, 45, n_samples)
    fico_range_low = rng.integers(580, 820, n_samples).astype(float)
    revol_bal = rng.integers(0, 50000, n_samples).astype(float)
    installment = loan_amnt / rng.integers(12, 60, n_samples)
    delinq_2yrs = rng.integers(0, 5, n_samples).astype(float)
    pub_rec = rng.integers(0, 3, n_samples).astype(float)

    logit = (
        -2.5
        + 0.00003 * loan_amnt
        - 0.00002 * annual_inc
        + 0.04 * dti
        - 0.008 * fico_range_low
        + 0.00001 * revol_bal
        + 0.15 * delinq_2yrs
        + 0.25 * pub_rec
    )
    prob = 1 / (1 + np.exp(-logit))
    target = (rng.random(n_samples) < prob).astype(int)

    return pd.DataFrame(
        {
            "loan_amnt": loan_amnt,
            "annual_inc": annual_inc,
            "dti": dti,
            "fico_range_low": fico_range_low,
            "revol_bal": revol_bal,
            "installment": installment,
            "delinq_2yrs": delinq_2yrs,
            "pub_rec": pub_rec,
            "target": target,
        }
    )


def seed_demo_models(force: bool = False) -> None:
    """Train and persist demo XGBoost artifacts if they do not exist."""
    model_path = MODELS_DIR / "xgboost_model.pkl"
    if model_path.exists() and not force:
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = _generate_synthetic_data()
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    save_pickle(model, model_path)
    save_pickle(scaler, MODELS_DIR / "scaler.pkl")
    save_pickle(
        {"label_encoders": {}, "feature_columns": FEATURE_COLUMNS},
        MODELS_DIR / "label_encoders.pkl",
    )

    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Demo models saved to {MODELS_DIR}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    seed_demo_models(force="--force" in sys.argv)
