"""
Feature engineering for credit risk — derived features, WOE / IV (scorecardpy), and selection.

Notes for portfolios
----------------------
**WOE (Weight of Evidence)** summarizes how often “goods” vs “bads” appear inside each bin
of a feature. It stabilizes sparse categories, linearizes non-linear effects, and is the
standard encoding in retail scorecards.

**IV (Information Value)** measures how much a feature separates good/bad overall; banks use
it as a quick screening metric (with sanity checks for “too high” IV).

**Leakage:** Binning + WOE are ideally fit on the training split only. This module follows
the common teaching pattern of fitting on the full processed file for simplicity; for
strict production, pass ``nrows`` for experiments or refactor to fit ``woebin`` on train.
"""

from __future__ import annotations

import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import scorecardpy as sc
except ImportError as exc:  # pragma: no cover - import guard for optional dep
    sc = None  # type: ignore[assignment]
    _SC_IMPORT_ERROR = exc
else:
    _SC_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_INPUT_CSV = PROCESSED_DATA_DIR / "processed_loan_data.csv"
FINAL_ENGINEERED_CSV = PROCESSED_DATA_DIR / "final_feature_engineered_data.csv"
SELECTED_FEATURES_CSV = PROCESSED_DATA_DIR / "selected_features.csv"
IV_VALUES_CSV = PROCESSED_DATA_DIR / "iv_values.csv"
WOE_BINS_PKL = PROCESSED_DATA_DIR / "woe_bins.pkl"
WOE_BINS_CSV = PROCESSED_DATA_DIR / "woe_bin_rules.csv"
RF_IMPORTANCE_CSV = PROCESSED_DATA_DIR / "rf_feature_importance.csv"

TARGET_COL = "loan_status"
LEGACY_TARGET_COL = "target"

IV_STRENGTH_RULES = """
IV interpretation (common industry cut-offs):
  < 0.02      weak (little predictive power)
  0.02–0.1    medium
  0.1–0.3     strong
  > 0.3       suspicious (often inspect data quality / leakage)
"""


def _ensure_src_on_path() -> None:
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _require_scorecardpy() -> Any:
    if sc is None:
        raise ImportError(
            "scorecardpy is required for WOE/IV. Install with: pip install scorecardpy"
        ) from _SC_IMPORT_ERROR
    return sc


def iv_predictive_strength(iv: float) -> str:
    """Map IV to a coarse strength label."""
    if iv < 0.02:
        return "weak"
    if iv < 0.1:
        return "medium"
    if iv < 0.3:
        return "strong"
    return "suspicious"


def load_data(
    path: Path | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """
    Load the processed loan dataset (post preprocessing).

    Parameters
    ----------
    path
        CSV path. Defaults to ``data/processed/processed_loan_data.csv``.
    **read_csv_kwargs
        Forwarded to :func:`pandas.read_csv` (e.g. ``nrows=`` for quick tests).
    """
    path = path or DEFAULT_INPUT_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    try:
        df = pd.read_csv(path, low_memory=False, **read_csv_kwargs)
    except Exception as exc:
        logger.exception("Failed reading CSV: %s", path)
        raise RuntimeError(f"Could not load processed data from {path}") from exc

    logger.info("Loaded processed data: %s rows, %s columns", len(df), df.shape[1])
    return df


def standardize_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the binary target is named ``loan_status`` (0 = Fully Paid, 1 = Charged Off).

    Accepts either ``loan_status`` or legacy ``target`` from the preprocessing export.
    """
    out = df.copy()
    if TARGET_COL in out.columns:
        return out
    if LEGACY_TARGET_COL in out.columns:
        out = out.rename(columns={LEGACY_TARGET_COL: TARGET_COL})
        logger.info("Renamed %s -> %s", LEGACY_TARGET_COL, TARGET_COL)
        return out
    raise KeyError(
        f"Expected target column {TARGET_COL!r} or {LEGACY_TARGET_COL!r}; "
        f"found columns: {list(out.columns)}"
    )


def analyze_feature_types(
    X: pd.DataFrame,
    exclude: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Split columns into numeric-like vs categorical-like (object/category).

    One-hot numeric dummies are treated as numerical.
    """
    ex = set(exclude or [])
    numerical: list[str] = []
    categorical: list[str] = []
    for col in X.columns:
        if col in ex:
            continue
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            numerical.append(col)
        else:
            categorical.append(col)
    return numerical, categorical


def create_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add credit-risk–style derived features.

    Ratios use ``loan_amnt``, ``annual_inc``, ``installment``, ``revol_bal`` when present.
    If ``annual_inc`` is zero or NaN, ratios are filled with the column median.

    **Note:** If ``processed_loan_data.csv`` contains *scaled* values (common after
    ``StandardScaler``), ratios are still valid ML features but are not literal
    dollar ratios. For literal ratios, engineer before scaling in preprocessing.

    **fico_category / income_category:** Uses FICO-style fixed cut-offs when the FICO
    column looks like real scores (300–850) and income looks like dollars; otherwise
    uses quantile bins with the same labels for robustness.
    """
    out = X.copy()
    eps = 1e-9

    def _safe_ratio(num: pd.Series, den: pd.Series, name: str) -> pd.Series:
        den_safe = den.replace(0, np.nan)
        r = num / (den_safe + eps)
        r = r.replace([np.inf, -np.inf], np.nan)
        med = r.median()
        if pd.isna(med):
            med = 0.0
        return r.fillna(med).astype(float).rename(name)

    if {"loan_amnt", "annual_inc"}.issubset(out.columns):
        out["loan_to_income_ratio"] = _safe_ratio(out["loan_amnt"], out["annual_inc"], "lti")
    if {"installment", "annual_inc"}.issubset(out.columns):
        out["installment_to_income_ratio"] = _safe_ratio(
            out["installment"], out["annual_inc"], "iti"
        )
    if {"revol_bal", "annual_inc"}.issubset(out.columns):
        out["revolving_to_income_ratio"] = _safe_ratio(out["revol_bal"], out["annual_inc"], "rti")

    if "fico_range_low" in out.columns:
        f = out["fico_range_low"]
        if f.dropna().quantile(0.9) > 200:  # looks like real FICO
            bins = [299, 579, 669, 739, 851]
            labels = ["Poor", "Fair", "Good", "Excellent"]
            out["fico_category"] = pd.cut(
                f, bins=bins, labels=labels, include_lowest=True
            ).astype(str)
        else:
            out["fico_category"] = pd.qcut(
                f,
                q=4,
                labels=["Poor", "Fair", "Good", "Excellent"],
                duplicates="drop",
            ).astype(str)

    if "annual_inc" in out.columns:
        inc = out["annual_inc"]
        if inc.max() > 10_000:  # looks like dollars
            out["income_category"] = pd.cut(
                inc,
                bins=[-1, 40_000, 80_000, inc.max() + 1],
                labels=["Low", "Medium", "High"],
            ).astype(str)
        else:
            out["income_category"] = pd.qcut(
                inc,
                q=3,
                labels=["Low", "Medium", "High"],
                duplicates="drop",
            ).astype(str)

    # scorecardpy handles strings; ensure categories are plain strings
    for col in ("fico_category", "income_category"):
        if col in out.columns:
            out[col] = out[col].replace("nan", "unknown").fillna("unknown").astype(str)

    return out


def perform_woe_binning(
    df: pd.DataFrame,
    y_col: str = TARGET_COL,
    stop_limit: float = 0.1,
    bin_num_limit: int = 8,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Fit automatic WOE binning (``scorecardpy.woebin``) and apply WOE encoding (``woebin_ply``).

    Parameters
    ----------
    df
        Must include the target column ``y_col`` (binary 0/1).
    y_col
        Name of the default flag (1 = bad).

    Returns
    -------
    bins
        Dictionary mapping feature name → binning table (scorecardpy structure).
    woe_df
        DataFrame of WOE values (column names typically end with ``_woe``).
    """
    s = _require_scorecardpy()
    if y_col not in df.columns:
        raise KeyError(f"Missing target column {y_col!r}")

    try:
        bins = s.woebin(
            df,
            y=y_col,
            print_step=0,
            no_cores=1,
            stop_limit=stop_limit,
            bin_num_limit=bin_num_limit,
            positive="bad|1",
        )
    except Exception as exc:
        logger.exception("woebin failed")
        raise RuntimeError("scorecardpy woebin could not complete") from exc

    X = df.drop(columns=[y_col])
    try:
        woe_df = s.woebin_ply(X, bins, print_step=0, no_cores=1)
    except Exception as exc:
        logger.exception("woebin_ply failed")
        raise RuntimeError("scorecardpy woebin_ply could not complete") from exc

    logger.info("WOE transform complete: %s rows, %s WOE columns", len(woe_df), woe_df.shape[1])
    return bins, woe_df


def calculate_iv(df: pd.DataFrame, y_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Compute IV per feature using ``scorecardpy.iv``.

    Returns a dataframe sorted by IV descending with columns:
    ``feature``, ``iv_score``, ``predictive_strength``.
    """
    s = _require_scorecardpy()
    if y_col not in df.columns:
        raise KeyError(f"Missing target column {y_col!r}")
    try:
        iv_raw = s.iv(df, y=y_col, x=None, order=True)
    except Exception as exc:
        logger.exception("IV calculation failed")
        raise RuntimeError("scorecardpy iv could not complete") from exc

    iv_raw = iv_raw.rename(columns={"variable": "feature"})
    if "info_value" in iv_raw.columns:
        iv_raw = iv_raw.rename(columns={"info_value": "iv_score"})
    elif "iv" in iv_raw.columns:
        iv_raw = iv_raw.rename(columns={"iv": "iv_score"})
    iv_raw["predictive_strength"] = iv_raw["iv_score"].map(iv_predictive_strength)
    iv_raw = iv_raw.sort_values("iv_score", ascending=False).reset_index(drop=True)
    return iv_raw


def select_features_iv(iv_table: pd.DataFrame, min_iv: float = 0.02) -> list[str]:
    """Keep feature names with IV strictly greater than ``min_iv``."""
    keep = iv_table.loc[iv_table["iv_score"] > min_iv, "feature"].tolist()
    logger.info("IV selection (>%s): %s features kept", min_iv, len(keep))
    return keep


def _woe_column_name(feature: str) -> str:
    return f"{feature}_woe"


def remove_correlated_features(
    woe_df: pd.DataFrame,
    iv_by_feature: dict[str, float],
    selected_features: list[str],
    corr_threshold: float = 0.9,
) -> list[str]:
    """
    Among WOE columns for ``selected_features``, drop redundant pairs with |corr| ≥ threshold.

    When two features are highly correlated, the one with **lower IV** is dropped
    (IV tie-breaks with lexicographic name for determinism).
    """
    woe_cols = []
    for f in selected_features:
        wc = _woe_column_name(f)
        if wc in woe_df.columns:
            woe_cols.append(wc)

    if len(woe_cols) < 2:
        return woe_cols

    sub = woe_df[woe_cols].copy()
    corr = sub.corr().abs()
    drop: set[str] = set()
    pairs: list[tuple[float, str, str]] = []
    for i, ci in enumerate(woe_cols):
        for j in range(i + 1, len(woe_cols)):
            cj = woe_cols[j]
            pairs.append((float(corr.loc[ci, cj]), ci, cj))
    pairs.sort(reverse=True)

    for ccorr, a, b in pairs:
        if ccorr < corr_threshold:
            break
        if a in drop or b in drop:
            continue
        ba, bb = a[:-4], b[:-4]
        iv_a = iv_by_feature.get(ba, 0.0)
        iv_b = iv_by_feature.get(bb, 0.0)
        if iv_a > iv_b or (iv_a == iv_b and ba < bb):
            drop.add(b)
        else:
            drop.add(a)

    kept = [c for c in woe_cols if c not in drop]
    logger.info(
        "Correlation filter (|r|>=%s): dropped %s WOE columns",
        corr_threshold,
        len(woe_cols) - len(kept),
    )
    return kept


def calculate_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train a small RandomForest and return feature importances sorted descending.

    ``X`` should be the WOE matrix (numeric only).
    """
    if X.shape[1] == 0:
        raise ValueError("No features passed to RandomForest.")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=50,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    try:
        rf.fit(X, y)
    except Exception as exc:
        logger.exception("RandomForest fit failed")
        raise RuntimeError("Could not fit RandomForest for feature importance") from exc

    imp = pd.DataFrame(
        {"feature": [c[:-4] if c.endswith("_woe") else c for c in X.columns], "rf_importance": rf.feature_importances_}
    ).sort_values("rf_importance", ascending=False)
    return imp.reset_index(drop=True)


def save_woe_bins(bins: dict[str, pd.DataFrame], pkl_path: Path, csv_path: Path) -> None:
    """Persist WOE binning rules as pickle and as a combined CSV for audit."""
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pkl_path.open("wb") as f:
            pickle.dump(bins, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        logger.exception("Failed to pickle WOE bins")
        raise RuntimeError(f"Could not save WOE bins to {pkl_path}") from exc

    try:
        parts: list[pd.DataFrame] = []
        for _var, tbl in bins.items():
            parts.append(tbl.copy())
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(csv_path, index=False)
    except Exception as exc:
        logger.exception("Failed to export WOE bin CSV")
        raise RuntimeError(f"Could not save WOE bin rules to {csv_path}") from exc

    logger.info("Saved WOE bins: %s and %s", pkl_path.name, csv_path.name)


def save_outputs(
    final_df: pd.DataFrame,
    selected_features: list[str],
    iv_table: pd.DataFrame,
    rf_importance: pd.DataFrame | None = None,
    *,
    final_csv: Path | None = None,
    selected_csv: Path | None = None,
    iv_csv: Path | None = None,
    rf_csv: Path | None = None,
) -> dict[str, Path]:
    """Write engineered dataset, selected feature list, IV table, and optional RF importances."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_csv = final_csv or FINAL_ENGINEERED_CSV
    selected_csv = selected_csv or SELECTED_FEATURES_CSV
    iv_csv = iv_csv or IV_VALUES_CSV
    rf_csv = rf_csv or RF_IMPORTANCE_CSV

    paths: dict[str, Path] = {}
    try:
        final_df.to_csv(final_csv, index=False)
        paths["final_engineered"] = final_csv
        pd.DataFrame({"feature": selected_features}).to_csv(selected_csv, index=False)
        paths["selected_features"] = selected_csv
        iv_table.to_csv(iv_csv, index=False)
        paths["iv_values"] = iv_csv
        if rf_importance is not None:
            rf_importance.to_csv(rf_csv, index=False)
            paths["rf_importance"] = rf_csv
    except Exception as exc:
        logger.exception("save_outputs failed")
        raise RuntimeError("Could not write one or more output files") from exc

    logger.info("Saved outputs under %s", PROCESSED_DATA_DIR)
    return paths


def feature_engineering_pipeline(
    input_csv: Path | None = None,
    min_iv: float = 0.02,
    corr_threshold: float = 0.9,
    woe_stop_limit: float = 0.1,
    **read_csv_kwargs: Any,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    End-to-end feature engineering on the processed dataset.

    Steps
    -----
    1. Load CSV and standardize target name to ``loan_status``.
    2. Split ``X`` / ``y``; add derived features on ``X``.
    3. Rebuild modeling frame with ``loan_status`` for scorecardpy.
    4. WOE binning + WOE transform.
    5. IV table + IV-based feature filter.
    6. Remove highly correlated WOE features (greedy, IV tie-break).
    7. RandomForest feature importance (for ranking / reporting).
    8. Save final WOE matrix + target, IV table, selected features, WOE bins, RF importances.

    Returns
    -------
    final_df
        Selected WOE columns plus ``loan_status``.
    selected_features
        Base feature names (without ``_woe`` suffix) kept after IV + correlation filtering.
    iv_table
        Full IV report for all input features (before correlation pruning).
    """
    _require_scorecardpy()

    df0 = load_data(input_csv, **read_csv_kwargs)
    df0 = standardize_target_column(df0)

    y = df0[TARGET_COL].astype(int)
    X = df0.drop(columns=[TARGET_COL])
    X = create_derived_features(X)

    modeling = X.copy()
    modeling[TARGET_COL] = y

    bins, woe_df = perform_woe_binning(modeling, y_col=TARGET_COL, stop_limit=woe_stop_limit)
    save_woe_bins(bins, WOE_BINS_PKL, WOE_BINS_CSV)

    iv_table = calculate_iv(modeling, y_col=TARGET_COL)
    iv_by = {row["feature"]: float(row["iv_score"]) for _, row in iv_table.iterrows()}

    iv_keep = select_features_iv(iv_table, min_iv=min_iv)
    woe_kept = remove_correlated_features(woe_df, iv_by, iv_keep, corr_threshold=corr_threshold)

    # Base names without _woe
    selected_base = sorted({c[:-4] for c in woe_kept if c.endswith("_woe")})

    X_rf = woe_df[woe_kept].copy()
    rf_imp = calculate_feature_importance(X_rf, y)

    final_df = X_rf.copy()
    final_df[TARGET_COL] = y.values

    save_outputs(
        final_df,
        selected_base,
        iv_table,
        rf_importance=rf_imp,
    )

    logger.info("Pipeline done. Final shape=%s, selected=%s features", final_df.shape, len(selected_base))
    return final_df, selected_base, iv_table


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for :func:`create_derived_features`."""
    return create_derived_features(df)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    _ensure_src_on_path()
    _configure_logging()
    try:
        final_df, selected, iv_df = feature_engineering_pipeline()
        print("Final shape:", final_df.shape)
        print("Selected features (n=%s):" % len(selected), selected[:25], "..." if len(selected) > 25 else "")
        print(iv_df.head(10).to_string(index=False))
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except ImportError as e:
        logger.error("%s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Feature engineering failed: %s", e)
        sys.exit(1)
