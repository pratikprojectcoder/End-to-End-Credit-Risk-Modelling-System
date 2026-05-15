"""Shared helpers for the credit risk pipeline."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preprocess import PROJECT_ROOT


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def processed_dataset_path(filename: str = "train.parquet") -> Path:
    return PROJECT_ROOT / "data" / "processed" / filename
