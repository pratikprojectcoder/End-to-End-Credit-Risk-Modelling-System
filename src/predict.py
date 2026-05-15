"""Batch or single-row prediction using saved artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preprocess import MODELS_DIR
from utils import load_pickle


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """Return default-risk probabilities for each row of ``df``."""
    model = load_pickle(MODELS_DIR / "xgboost_model.pkl")
    scaler = load_pickle(MODELS_DIR / "scaler.pkl")
    encoders = load_pickle(MODELS_DIR / "label_encoders.pkl")
    if model is None:
        raise RuntimeError("Model is not trained yet.")

    feature_cols = encoders.get("feature_columns", [])
    X = df.reindex(columns=feature_cols, fill_value=0).select_dtypes(include=[np.number]).fillna(0)
    Xs = scaler.transform(X)
    return model.predict_proba(Xs)[:, 1]
