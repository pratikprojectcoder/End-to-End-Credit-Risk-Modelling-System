"""Batch or single-row prediction using saved artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inference_service import get_inference_service


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """Return default-risk probabilities for each row of ``df``."""
    service = get_inference_service()
    return service.predict_proba(df)
