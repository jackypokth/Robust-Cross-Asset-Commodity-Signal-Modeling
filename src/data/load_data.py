"""
src/data/load_data.py
---------------------
Clean data loading functions for the competition dataset.

Design principles:
  - No feature engineering here — loading only.
  - All paths resolved from src/utils/paths.py.
  - Functions return typed DataFrames; callers handle downstream processing.
  - Missing-data issues are flagged with warnings, not silently fixed.

Data layout (confirmed against downloaded files):
  - train.csv         : 1961 rows × 558 cols  (date_id + 557 features)
  - train_labels.csv  : 1961 rows × 425 cols  (date_id + 424 targets)
  - test.csv          : 134  rows × 559 cols  (date_id + 558 features)
  - target_pairs.csv  : 424  rows × 3 cols    (target, lag, pair)
  - test_labels_lag_N : 134  rows × ~108 cols (date_id + 107 targets per lag)

  date_id is an integer day index (0-based), not a calendar date.
  All data is already in wide format — no pivoting required.

Usage:
    from src.data.load_data import load_train, load_labels, load_test
    from src.data.load_data import load_target_pairs, load_test_labels
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from src.utils.paths import (
    LABELS_FILE,
    TARGET_PAIRS_FILE,
    TEST_FILE,
    TEST_LABELS_LAG,
    TRAIN_FILE,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_csv_safe(path: Path, description: str) -> pd.DataFrame:
    """Read a CSV file with basic validation.

    Args:
        path: Absolute path to the CSV file.
        description: Human-readable name used in error messages.

    Returns:
        Raw DataFrame as loaded by pandas.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at: {path}\n"
            "Download competition data from Kaggle and place it in data/raw/."
        )
    return pd.read_csv(path)


def _warn_missing(df: pd.DataFrame, context: str) -> None:
    """Emit a warning if df contains any NaN values."""
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        warnings.warn(
            f"{context}: {n_missing:,} missing values. "
            "Inspect df.isnull().sum() to understand the pattern.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Public loading functions
# ---------------------------------------------------------------------------

def load_train(
    path: Path | None = None,
) -> pd.DataFrame:
    """Load the training feature matrix.

    Returns a wide-format DataFrame: rows are trading days (date_id),
    columns are feature names (commodity prices, FX rates, etc.).

    Args:
        path: Override the default path (useful for testing with samples).

    Returns:
        DataFrame of shape (1961, 558). Index is the default integer index;
        date_id is a regular column. Feature values are floats; expect
        ~45k missing entries due to exchange calendar mismatches.

    Raises:
        FileNotFoundError: If train.csv is not present in data/raw/.
    """
    file_path = path or TRAIN_FILE
    df = _read_csv_safe(file_path, "Training features")

    n_rows, n_cols = df.shape
    print(f"[load_train] {n_rows:,} rows × {n_cols} cols")
    _warn_missing(df, "load_train")

    return df


def load_labels(
    path: Path | None = None,
) -> pd.DataFrame:
    """Load the training labels.

    Returns a wide-format DataFrame: rows are trading days (date_id),
    columns are target_0 through target_423 (424 targets total).

    Each target is the return of an asset or asset-pair at a given lag.
    Use load_target_pairs() to map target names to (lag, pair) descriptions.

    Args:
        path: Override the default path.

    Returns:
        DataFrame of shape (1961, 425). First column is date_id (int).

    Raises:
        FileNotFoundError: If train_labels.csv is not present in data/raw/.
    """
    file_path = path or LABELS_FILE
    df = _read_csv_safe(file_path, "Training labels")

    n_rows, n_cols = df.shape
    n_targets = n_cols - 1  # subtract date_id
    print(f"[load_labels] {n_rows:,} rows × {n_targets} targets")
    _warn_missing(df, "load_labels")

    return df


def load_test(
    path: Path | None = None,
) -> pd.DataFrame:
    """Load the test feature matrix.

    Returns a wide-format DataFrame covering the held-out evaluation period
    (date_id 1827–1960, 134 trading days).

    Args:
        path: Override the default path.

    Returns:
        DataFrame of shape (134, 559). No target columns are present.

    Raises:
        FileNotFoundError: If test.csv is not present in data/raw/.
    """
    file_path = path or TEST_FILE
    df = _read_csv_safe(file_path, "Test features")

    n_rows, n_cols = df.shape
    print(f"[load_test] {n_rows:,} rows × {n_cols} cols")

    return df


def load_target_pairs(
    path: Path | None = None,
) -> pd.DataFrame:
    """Load the target-to-pair mapping.

    Maps each of the 424 target column names to a forecast lag (1–4) and
    a human-readable pair description (e.g. "LME_AH_Close - LME_ZS_Close").
    Single-asset targets have no " - " separator in the pair string.

    Args:
        path: Override the default path.

    Returns:
        DataFrame of shape (424, 3) with columns: target, lag, pair.
        - target (str): e.g. "target_0"
        - lag (int): forecast horizon in trading days (1, 2, 3, or 4)
        - pair (str): e.g. "LME_AH_Close - LME_ZS_Close"

    Raises:
        FileNotFoundError: If target_pairs.csv is not present in data/raw/.
    """
    file_path = path or TARGET_PAIRS_FILE
    df = _read_csv_safe(file_path, "Target pairs metadata")

    print(f"[load_target_pairs] {len(df)} targets "
          f"({df['lag'].value_counts().sort_index().to_dict()})")

    return df


def load_test_labels(
    lag: int,
    path: Path | None = None,
) -> pd.DataFrame:
    """Load the held-out test labels for a given forecast lag.

    These files are provided for local evaluation after the competition.
    Each file contains the realized targets for the test period at a
    specific forecast horizon.

    Args:
        lag: Forecast horizon (1, 2, 3, or 4 trading days).
        path: Override the default path.

    Returns:
        DataFrame with columns: date_id, target_0, ..., target_N.
        The date_id values are shifted forward by `lag` relative to the
        test feature date_ids.

    Raises:
        ValueError: If lag is not in {1, 2, 3, 4}.
        FileNotFoundError: If the file is not present in data/raw/.
    """
    if lag not in (1, 2, 3, 4):
        raise ValueError(f"lag must be 1, 2, 3, or 4; got {lag}")

    file_path = path or TEST_LABELS_LAG[lag]
    df = _read_csv_safe(file_path, f"Test labels (lag={lag})")

    n_rows, n_cols = df.shape
    print(f"[load_test_labels lag={lag}] {n_rows:,} rows × {n_cols - 1} targets")

    return df


# ---------------------------------------------------------------------------
# Convenience: extract a single lag from wide labels
# ---------------------------------------------------------------------------

def get_labels_for_lag(
    labels: pd.DataFrame,
    pairs: pd.DataFrame,
    lag: int,
) -> pd.DataFrame:
    """Extract the subset of training labels for a specific forecast lag.

    Args:
        labels: Wide-format labels DataFrame from load_labels().
        pairs: Target pairs DataFrame from load_target_pairs().
        lag: Forecast horizon (1, 2, 3, or 4).

    Returns:
        Wide-format DataFrame with date_id and only the target columns
        belonging to the requested lag. Column order matches target_pairs.csv.

    Example:
        >>> labels = load_labels()
        >>> pairs  = load_target_pairs()
        >>> lag1   = get_labels_for_lag(labels, pairs, lag=1)
        >>> lag1.shape  # (1961, 107)
    """
    if lag not in (1, 2, 3, 4):
        raise ValueError(f"lag must be 1, 2, 3, or 4; got {lag}")

    lag_targets = pairs.loc[pairs["lag"] == lag, "target"].tolist()
    return labels[["date_id"] + lag_targets].copy()
