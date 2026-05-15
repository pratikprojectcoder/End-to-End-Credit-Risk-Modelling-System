"""Model explanations (feature importance and optional SHAP)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preprocess import MODELS_DIR
from utils import load_pickle


def feature_importance() -> dict[str, float]:
    """Return XGBoost ``feature_importances_`` mapped to column names."""
    model = load_pickle(MODELS_DIR / "xgboost_model.pkl")
    encoders = load_pickle(MODELS_DIR / "label_encoders.pkl")
    if model is None:
        raise RuntimeError("Model is not trained yet.")

    names = list(encoders.get("feature_columns", []))
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {}
    if len(names) != len(importances):
        names = [f"f{i}" for i in range(len(importances))]
    return dict(sorted(zip(names, importances.tolist()), key=lambda x: -x[1]))


def shap_summary_plot(**kwargs: Any) -> None:
    """Optional SHAP summary plot; requires ``pip install shap``."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as e:
        raise ImportError("Install shap and matplotlib to use shap_summary_plot.") from e

    model = load_pickle(MODELS_DIR / "xgboost_model.pkl")
    scaler = load_pickle(MODELS_DIR / "scaler.pkl")
    encoders = load_pickle(MODELS_DIR / "label_encoders.pkl")
    X = kwargs.get("X")
    if X is None:
        raise ValueError("Pass feature matrix as X=... for SHAP.")

    feature_cols = encoders.get("feature_columns", [])
    X = X.reindex(columns=feature_cols, fill_value=0)
    Xs = scaler.transform(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)
    shap.summary_plot(shap_values, Xs, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for name, score in list(feature_importance().items())[:15]:
        print(f"{name}: {score:.4f}")
