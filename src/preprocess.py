"""Credit risk dataset preprocessing — modular, production-style pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths (project root = parent of ``src/``)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_RAW_FILENAMES: tuple[str, ...] = (
    "accepted_2007_to_2018.csv",
    "accepted_2007_to_2018Q4.csv",
)
PROCESSED_CSV_NAME = "processed_loan_data.csv"
SCALER_JOBLIB_NAME = "feature_scaler.joblib"

IMPORTANT_COLUMNS: list[str] = [
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "purpose",
    "dti",
    "fico_range_low",
    "installment",
    "revol_bal",
    "delinq_2yrs",
    "pub_rec",
    "loan_status",
]

IQR_OUTLIER_COLUMNS: list[str] = ["annual_inc", "loan_amnt", "dti", "revol_bal"]

EMP_LENGTH_MAP: dict[str, int] = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}

logger = logging.getLogger(__name__)


def _ensure_src_on_path() -> None:
    """Allow running this file as a script: ``python src/preprocess.py``."""
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _resolve_raw_path(raw_path: Path | None) -> Path:
    """Pick the first existing raw CSV from defaults or the given path."""
    if raw_path is not None:
        if not raw_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {raw_path}")
        return raw_path

    for name in DEFAULT_RAW_FILENAMES:
        candidate = RAW_DATA_DIR / name
        if candidate.is_file():
            logger.info("Using raw dataset: %s", candidate)
            return candidate

    tried = ", ".join(str(RAW_DATA_DIR / n) for n in DEFAULT_RAW_FILENAMES)
    raise FileNotFoundError(
        f"No raw dataset found under {RAW_DATA_DIR}. Tried: {tried}. "
        "Place accepted_2007_to_2018.csv there or pass raw_path= explicitly."
    )


def load_data(raw_path: Path | None = None, **read_csv_kwargs: Any) -> pd.DataFrame:
    """
    Load the Lending Club-style accepted loans CSV.

    Parameters
    ----------
    raw_path
        Full path to the CSV. If ``None``, uses ``DEFAULT_RAW_FILENAMES`` under
        ``data/raw/`` (first file that exists).
    **read_csv_kwargs
        Extra arguments forwarded to :func:`pandas.read_csv` (e.g. ``nrows`` for tests).

    Returns
    -------
    pandas.DataFrame
    """
    path = _resolve_raw_path(raw_path)
    try:
        df = pd.read_csv(path, low_memory=False, **read_csv_kwargs)
    except Exception as exc:
        logger.exception("Failed to read CSV: %s", path)
        raise RuntimeError(f"Could not read dataset at {path}") from exc

    logger.info("Loaded %s rows and %s columns from %s", len(df), df.shape[1], path.name)
    return df


def select_features(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Keep only the modeling columns defined for this project.

    Missing columns raise ``KeyError`` so misnamed files fail fast.
    """
    cols = list(columns) if columns is not None else IMPORTANT_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataframe: {missing}")

    out = df.loc[:, cols].copy()
    logger.info("Selected %s feature columns (+ loan_status).", len(cols))
    return out


def handle_target(
    df: pd.DataFrame,
    status_col: str = "loan_status",
    positive_label: str = "Charged Off",
    negative_label: str = "Fully Paid",
) -> pd.DataFrame:
    """
    Restrict to binary outcomes and map to 0/1.

    - Fully Paid → 0
    - Charged Off → 1

    The original ``loan_status`` column is dropped; the target is ``target``.
    """
    if status_col not in df.columns:
        raise KeyError(f"Missing status column: {status_col!r}")

    allowed = {negative_label, positive_label}
    out = df[df[status_col].isin(allowed)].copy()
    before = len(df)
    after = len(out)
    logger.info(
        "Target filter: kept %s of %s rows (binary statuses only).",
        after,
        before,
    )

    mapping = {negative_label: 0, positive_label: 1}
    out["target"] = out[status_col].map(mapping)
    out = out.drop(columns=[status_col])
    return out


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values: median for numeric columns, mode for non-numeric.

    If a column has no mode (all NaN), it is filled with 0 for numeric or
    the string ``\"unknown\"`` for non-numeric after coercion attempt.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            med = out[col].median()
            if pd.isna(med):
                out[col] = out[col].fillna(0)
            else:
                out[col] = out[col].fillna(med)
        else:
            mode = out[col].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "unknown"
            out[col] = out[col].fillna(fill)
    return out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and reset the index."""
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %s duplicate rows.", before - len(out))
    return out


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize text-derived fields.

    - ``term``: ``\" 36 months\"`` → ``36`` (integer months).
    - ``int_rate``: ``\"10.5%\"`` → ``10.5`` (float).
    - ``emp_length``: maps categories to integers 0–10 (years bucket).
    """
    out = df.copy()

    if "term" in out.columns:
        out["term"] = (
            out["term"]
            .astype(str)
            .str.strip()
            .str.replace(" months", "", regex=False)
            .replace("", np.nan)
        )
        out["term"] = pd.to_numeric(out["term"], errors="coerce")

    if "int_rate" in out.columns:
        out["int_rate"] = (
            out["int_rate"].astype(str).str.replace("%", "", regex=False).str.strip()
        )
        out["int_rate"] = pd.to_numeric(out["int_rate"], errors="coerce")

    if "emp_length" in out.columns:
        out["emp_length"] = out["emp_length"].map(EMP_LENGTH_MAP)

    # Any coercion NaNs (unknown categories) → median for numeric
    num_cols = out.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if out[col].isna().any():
            med = out[col].median()
            out[col] = out[col].fillna(0 if pd.isna(med) else med)

    return out


def remove_outliers(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Winsorize numeric columns using the IQR rule (clip to ``[Q1 - k*IQR, Q3 + k*IQR]``).

    Applied by default to ``annual_inc``, ``loan_amnt``, ``dti``, ``revol_bal``.
    """
    cols = list(columns) if columns is not None else IQR_OUTLIER_COLUMNS
    out = df.copy()
    for col in cols:
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - factor * iqr
        high = q3 + factor * iqr
        out[col] = out[col].clip(lower=low, upper=high)
    return out


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categoricals:

    - ``grade``: :class:`sklearn.preprocessing.LabelEncoder`
    - ``home_ownership``, ``purpose``: one-hot (pandas ``get_dummies``)
    """
    out = df.copy()

    if "grade" in out.columns:
        le = LabelEncoder()
        out["grade"] = le.fit_transform(out["grade"].astype(str))

    ohe_cols = [c for c in ("home_ownership", "purpose") if c in out.columns]
    if ohe_cols:
        out = pd.get_dummies(out, columns=ohe_cols, drop_first=False, dtype=int)

    # Ensure purely numeric matrix for scaling
    bool_cols = out.select_dtypes(include=["bool"]).columns
    for c in bool_cols:
        out[c] = out[c].astype(int)

    return out


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], StandardScaler, list[str]]:
    """
    Stratified train/test split and :class:`StandardScaler` fit on the train set.

    Returns
    -------
    (X_train, X_test, y_train, y_test), scaler, feature_names
        Arrays are float64; ``y_*`` are shape ``(n,)`` with values ``{0,1}``.
    """
    if "target" in X.columns:
        raise ValueError("Pass features only; ``target`` should not be in X.")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_tr)
    X_test = scaler.transform(X_te)

    y_train = y_tr.to_numpy().astype(int)
    y_test = y_te.to_numpy().astype(int)

    logger.info(
        "Train shape: %s, Test shape: %s, num features: %s",
        X_train.shape,
        X_test.shape,
        len(feature_names),
    )
    return (X_train, X_test, y_train, y_test), scaler, feature_names


def save_processed_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    csv_path: Path | None = None,
    scaler_path: Path | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[Path, Path]:
    """
    Save the scaled feature matrix (train + test rows) with ``target`` to CSV,
    and persist the fitted ``StandardScaler`` with joblib.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = csv_path or (PROCESSED_DATA_DIR / PROCESSED_CSV_NAME)
    scaler_path = scaler_path or (PROCESSED_DATA_DIR / SCALER_JOBLIB_NAME)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    full = pd.DataFrame(X_all, columns=feature_names)
    full["target"] = y_all

    try:
        full.to_csv(csv_path, index=False)
    except Exception as exc:
        logger.exception("Failed to write CSV: %s", csv_path)
        raise RuntimeError(f"Could not save processed CSV to {csv_path}") from exc

    if scaler is None:
        raise ValueError("``scaler`` is required to save with joblib.")

    try:
        joblib.dump(scaler, scaler_path)
    except Exception as exc:
        logger.exception("Failed to write scaler: %s", scaler_path)
        raise RuntimeError(f"Could not save scaler to {scaler_path}") from exc

    logger.info("Saved processed data to %s", csv_path)
    logger.info("Saved scaler to %s", scaler_path)
    return csv_path, scaler_path


def preprocess_pipeline(
    raw_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    output_csv: Path | None = None,
    scaler_output: Path | None = None,
    **read_csv_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full preprocessing pipeline end-to-end.

    Steps
    -----
    1. Load CSV
    2. Select project columns
    3. Binary target mapping
    4. Impute missing values (median / mode)
    5. Drop duplicates
    6. Clean ``term``, ``int_rate``, ``emp_length``
    7. IQR winsorization on selected numerics
    8. Encode ``grade`` + one-hot ``home_ownership`` / ``purpose``
    9. Stratified split + ``StandardScaler`` on train
    10. Save ``processed_loan_data.csv`` and ``feature_scaler.joblib``

    Returns
    -------
    X_train, X_test, y_train, y_test
        Scaled feature matrices and binary targets.
    """
    df = load_data(raw_path=raw_path, **read_csv_kwargs)
    df = select_features(df)
    df = handle_target(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = clean_columns(df)
    df = remove_outliers(df)
    df = encode_features(df)

    if "target" not in df.columns:
        raise RuntimeError("``target`` column missing after preprocessing.")

    y = df["target"]
    X = df.drop(columns=["target"])

    (X_train, X_test, y_train, y_test), scaler, names = split_and_scale(
        X, y, test_size=test_size, random_state=random_state
    )

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names=names,
        csv_path=output_csv,
        scaler_path=scaler_output,
        scaler=scaler,
    )

    logger.info("Final combined shape (rows, cols): %s", (len(y), X.shape[1]))
    return X_train, X_test, y_train, y_test


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    _ensure_src_on_path()
    _configure_logging()
    try:
        Xt_tr, Xt_te, yt_tr, yt_te = preprocess_pipeline()
        print("X_train:", Xt_tr.shape, "X_test:", Xt_te.shape)
        print("y_train:", yt_tr.shape, "y_test:", yt_te.shape)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Preprocessing failed: %s", e)
        sys.exit(1)
