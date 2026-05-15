"""Hyperparameter tuning using Optuna for XGBoost."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
# LOAD DATA
# =========================================================

processed_path = (
    PROCESSED_DATA_DIR /
    "balanced_train_data.csv"
)

df = pd.read_csv(processed_path)

target_column = "loan_status"

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


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# =========================================================
# OPTUNA OBJECTIVE FUNCTION
# =========================================================

def objective(trial):
    """
    Objective function for Optuna optimization.
    """

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            50,
            300
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.5,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0
        ),

        "random_state": 42,

        "eval_metric": "logloss",
    }

    model = XGBClassifier(**params)

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)

    return auc


# =========================================================
# RUN OPTUNA STUDY
# =========================================================

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=20
)


# =========================================================
# BEST PARAMETERS
# =========================================================

print("\nBest Parameters:")
print(study.best_params)

print("\nBest ROC-AUC Score:")
print(study.best_value)


# =========================================================
# TRAIN FINAL MODEL
# =========================================================

best_model = XGBClassifier(
    **study.best_params,
    random_state=42,
    eval_metric="logloss",
)

best_model.fit(X_train, y_train)


# =========================================================
# SAVE MODEL
# =========================================================

save_pickle(
    best_model,
    MODELS_DIR / "tuned_xgboost_model.pkl"
)

print("\nTuned model saved successfully.")