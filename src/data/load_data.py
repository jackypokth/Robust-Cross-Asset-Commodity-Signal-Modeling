"""
src/data/load_data.py
---------------------
Clean data loading functions for the competition dataset.

Design principles:
  - No feature engineering here — loading only.
  - All paths resolved from src/utils/paths.py.
  - Functions return typed DataFrames; callers handle downstream processing.
  - Missing-data and dtype issues are flagged with warnings, not silently fixed.

Usage:
    from src.data.load_data import load_train, load_test, load_sample_submission
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from src.utils.paths import SAMPLE_SUBMISSION_FILE, TEST_FILE, TRAIN_FILE


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


def _parse_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Parse the date column to datetime if it exists.

    Args:
        df: Input DataFrame.
        date_col: Name of the date column.

    Returns:
        DataFrame with date_col converted to datetime (in-place copy).
    """
    if date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def load_train(
    date_col: str = "date",
    parse_dates: bool = True,
    path: Path | None = None,
) -> pd.DataFrame:
    """Load the training dataset.

    Args:
        date_col: Name of the column containing observation dates.
            Defaults to "date" — update if the actual column name differs.
        parse_dates: If True, convert the date column to datetime.
        path: Override the default path (useful for testing with sample files).

    Returns:
        DataFrame with all columns from train.csv. Date column is datetime
        if parse_dates=True.

    Raises:
        FileNotFoundError: If train.csv is not present in data/raw/.

    Notes:
        - The target column name may vary; inspect df.columns after loading.
        - No NaN imputation is performed here.
    """
    file_path = path or TRAIN_FILE
    df = _read_csv_safe(file_path, "Training data")

    if parse_dates:
        df = _parse_dates(df, date_col=date_col)

    n_rows, n_cols = df.shape
    n_missing = df.isnull().sum().sum()

    print(f"[load_train] Loaded {n_rows:,} rows × {n_cols} columns")
    if n_missing > 0:
        warnings.warn(
            f"Training data contains {n_missing:,} missing values. "
            "Inspect df.isnull().sum() to understand the pattern.",
            UserWarning,
            stacklevel=2,
        )

    return df


def load_test(
    date_col: str = "date",
    parse_dates: bool = True,
    path: Path | None = None,
) -> pd.DataFrame:
    """Load the test dataset.

    Args:
        date_col: Name of the date column.
        parse_dates: If True, convert the date column to datetime.
        path: Override the default path (useful for testing).

    Returns:
        DataFrame with all columns from test.csv (no target column).

    Raises:
        FileNotFoundError: If test.csv is not present in data/raw/.
    """
    file_path = path or TEST_FILE
    df = _read_csv_safe(file_path, "Test data")

    if parse_dates:
        df = _parse_dates(df, date_col=date_col)

    n_rows, n_cols = df.shape
    print(f"[load_test] Loaded {n_rows:,} rows × {n_cols} columns")

    return df


def load_sample_submission(
    path: Path | None = None,
) -> pd.DataFrame:
    """Load the sample submission file.

    Useful for understanding the required submission format.

    Args:
        path: Override the default path.

    Returns:
        DataFrame matching the submission format (id + target columns).

    Raises:
        FileNotFoundError: If sample_submission.csv is not in data/raw/.
    """
    file_path = path or SAMPLE_SUBMISSION_FILE
    df = _read_csv_safe(file_path, "Sample submission")
    print(f"[load_sample_submission] Columns: {list(df.columns)}")
    return df


def pivot_to_wide(
    df: pd.DataFrame,
    date_col: str = "date",
    commodity_col: str = "commodity",
    value_col: str = "target",
) -> pd.DataFrame:
    """Pivot long-format data to wide format (dates × commodities).

    The competition metric expects predictions in wide format:
    rows = time steps, columns = commodities.

    Args:
        df: Long-format DataFrame with date, commodity, and value columns.
        date_col: Name of the date column.
        commodity_col: Name of the commodity identifier column.
        value_col: Name of the value column to pivot.

    Returns:
        Wide-format DataFrame with shape (T, N) where T = unique dates
        and N = unique commodities. Index is the date column.

    Example:
        >>> wide_train = pivot_to_wide(train_df, value_col="target")
        >>> wide_pred = pivot_to_wide(pred_df, value_col="prediction")
        >>> score = mean_row_pearson(wide_pred, wide_train)
    """
    wide = df.pivot(index=date_col, columns=commodity_col, values=value_col)
    wide.columns.name = None
    return wide
