"""
src/evaluation/metric.py
------------------------
Local implementation of the competition evaluation metric.

The competition uses **mean row-wise Pearson correlation**:
  - For each time step, compute Pearson correlation between predicted and
    actual returns across all commodities.
  - Average those per-period correlations.

This is a cross-sectional ranking metric: scale and mean shift of predictions
within a time period do not affect the score. Only the relative ordering
of predicted returns across commodities at each time step matters.

References:
    - docs/metric_explainer.md  — plain-English explanation
    - tests/test_metric.py      — unit tests with synthetic examples
"""

from __future__ import annotations

import warnings
from typing import Literal  # Literal stays; available natively in 3.8+

import numpy as np
import pandas as pd


def pearson_corr_row(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Pearson correlation between two 1-D arrays.

    Args:
        y_pred: Predicted values for a single time step, shape (n_assets,).
        y_true: Actual values for a single time step, shape (n_assets,).

    Returns:
        Pearson correlation coefficient in [-1, 1], or NaN if either input
        has zero variance (constant predictions or targets).

    Notes:
        NaN values in either array are excluded pairwise from the calculation.
    """
    if len(y_pred) != len(y_true):
        raise ValueError(
            f"y_pred and y_true must have the same length. "
            f"Got {len(y_pred)} and {len(y_true)}."
        )

    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    y_pred_clean = y_pred[mask]
    y_true_clean = y_true[mask]

    if len(y_pred_clean) < 2:
        return float("nan")

    std_pred = y_pred_clean.std()
    std_true = y_true_clean.std()

    if std_pred == 0.0 or std_true == 0.0:
        return float("nan")

    corr_matrix = np.corrcoef(y_pred_clean, y_true_clean)
    return float(corr_matrix[0, 1])


def mean_row_pearson(
    y_pred: pd.DataFrame,
    y_true: pd.DataFrame,
    nan_policy: Literal["warn", "raise", "ignore"] = "warn",
) -> float:
    """Mean row-wise Pearson correlation — the competition metric.

    Computes Pearson correlation between predicted and actual values for each
    row (time step), then averages across all rows. Rows where the correlation
    is NaN (e.g., due to constant predictions) are excluded from the average
    with an optional warning.

    Args:
        y_pred: DataFrame of shape (T, N) where T is the number of time steps
            and N is the number of assets/commodities. Index should be dates
            or time-step identifiers. Columns should be commodity identifiers.
        y_true: DataFrame of the same shape and column order as y_pred.
        nan_policy: How to handle NaN correlation values per row.
            - "warn"  : Warn and exclude NaN rows (default, recommended).
            - "raise" : Raise ValueError if any NaN correlations are found.
            - "ignore": Silently exclude NaN rows.

    Returns:
        Scalar float: the mean Pearson correlation across all valid time steps.
        Returns NaN if all time steps produce NaN correlations.

    Raises:
        ValueError: If y_pred and y_true do not have matching shapes or columns,
            or if nan_policy="raise" and NaN correlations are encountered.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> pred = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]})
        >>> true = pd.DataFrame({"A": [0.1, 0.2, 0.3], "B": [0.3, 0.2, 0.1]})
        >>> mean_row_pearson(pred, true)
        1.0
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred and y_true must have identical shapes. "
            f"Got {y_pred.shape} and {y_true.shape}."
        )
    if list(y_pred.columns) != list(y_true.columns):
        raise ValueError(
            "y_pred and y_true must have identical column names (commodity IDs). "
            "Reorder or rename columns before calling this function."
        )

    corrs: list[float] = []

    for t in range(len(y_pred)):
        row_pred = y_pred.iloc[t].to_numpy(dtype=float)
        row_true = y_true.iloc[t].to_numpy(dtype=float)
        c = pearson_corr_row(row_pred, row_true)
        corrs.append(c)

    corr_array = np.array(corrs, dtype=float)
    nan_mask = np.isnan(corr_array)
    n_nan = int(nan_mask.sum())

    if n_nan > 0:
        message = (
            f"{n_nan} of {len(corr_array)} time steps produced NaN correlation "
            "(likely due to constant predictions or fewer than 2 valid values). "
            "These rows are excluded from the mean."
        )
        if nan_policy == "raise":
            raise ValueError(message)
        elif nan_policy == "warn":
            warnings.warn(message, UserWarning, stacklevel=2)

    valid_corrs = corr_array[~nan_mask]

    if len(valid_corrs) == 0:
        warnings.warn(
            "All time steps produced NaN correlation. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    return float(np.mean(valid_corrs))


def per_period_pearson(
    y_pred: pd.DataFrame,
    y_true: pd.DataFrame,
) -> pd.Series:
    """Compute Pearson correlation for each time step, returned as a Series.

    Useful for diagnosing which time periods the model struggles with.

    Args:
        y_pred: DataFrame of shape (T, N).
        y_true: DataFrame of shape (T, N).

    Returns:
        pd.Series of length T with the per-period correlation values.
        Index matches y_pred.index. NaN where correlation is undefined.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}."
        )

    corrs = [
        pearson_corr_row(
            y_pred.iloc[t].to_numpy(dtype=float),
            y_true.iloc[t].to_numpy(dtype=float),
        )
        for t in range(len(y_pred))
    ]

    return pd.Series(corrs, index=y_pred.index, name="pearson_corr")
