"""
src/evaluation/metric.py
------------------------
Evaluation metric implementations for the competition.

Competition metric (official):
    spearman_sharpe_score — mean row-wise Spearman correlation divided by its
    standard deviation across time steps. This is a Sharpe-like ratio that
    rewards consistent cross-sectional ranking, not just average performance.

Offline proxy metric:
    mean_row_pearson — mean row-wise Pearson correlation. Faster to compute
    and useful for sanity checks, but NOT the official competition metric.
    Pearson and Spearman agree closely when the prediction distribution is
    approximately monotone, but diverge near outliers. Always report both.

References:
    - docs/metric_explainer.md  — explanation of both metrics
    - tests/test_metric.py      — unit tests with synthetic examples
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# Official competition metric — Spearman-Sharpe score
# ============================================================================

def spearman_corr_row(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Spearman rank correlation between two 1-D arrays.

    Rank-transforms both arrays (handling ties by average) and computes
    the Pearson correlation of the ranks. NaN values are excluded pairwise.

    Args:
        y_pred: Predicted values for a single time step, shape (n_assets,).
        y_true: Actual values for a single time step, shape (n_assets,).

    Returns:
        Spearman correlation in [-1, 1], or NaN if fewer than 2 valid pairs
        remain after removing NaNs, or if either array has zero variance.

    Raises:
        ValueError: If y_pred and y_true have different lengths.
    """
    if len(y_pred) != len(y_true):
        raise ValueError(
            f"y_pred and y_true must have the same length. "
            f"Got {len(y_pred)} and {len(y_true)}."
        )

    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() < 2:
        return float("nan")

    result = stats.spearmanr(y_pred[mask], y_true[mask])
    return float(result.statistic)


def spearman_sharpe_score(
    y_pred: pd.DataFrame,
    y_true: pd.DataFrame,
    ddof: int = 1,
    nan_policy: Literal["warn", "raise", "ignore"] = "warn",
) -> float:
    """Official competition metric: mean Spearman correlation / std.

    For each row (time step), compute the Spearman rank correlation between
    predicted and actual values across all targets. Then return the mean of
    those per-period correlations divided by their standard deviation.

    This Sharpe-like ratio penalises inconsistency: a model that scores 0.3
    every period is rated higher than one that scores 0.8 half the time and
    -0.2 the other half, even though the latter has a higher mean.

    Args:
        y_pred: DataFrame of shape (T, N) — T time steps, N targets.
            Index should be date_id or time-step identifiers.
        y_true: DataFrame of the same shape and column order as y_pred.
        ddof: Delta degrees of freedom for the standard deviation.
            Use ddof=1 (default, sample std) to match a standard Sharpe ratio.
        nan_policy: How to handle time steps where correlation is NaN.
            - "warn"  : Warn and exclude (default).
            - "raise" : Raise ValueError.
            - "ignore": Silently exclude.

    Returns:
        Scalar float: mean(spearman_r_t) / std(spearman_r_t).
        Returns NaN if fewer than 2 valid time steps remain, or if std is 0.

    Raises:
        ValueError: If shapes or columns do not match, or nan_policy="raise"
            and NaN correlations are found.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred and y_true must have identical shapes. "
            f"Got {y_pred.shape} and {y_true.shape}."
        )
    if list(y_pred.columns) != list(y_true.columns):
        raise ValueError(
            "y_pred and y_true must have identical column names. "
            "Reorder or rename columns before calling this function."
        )

    corrs: list[float] = []
    for t in range(len(y_pred)):
        row_pred = y_pred.iloc[t].to_numpy(dtype=float)
        row_true = y_true.iloc[t].to_numpy(dtype=float)
        corrs.append(spearman_corr_row(row_pred, row_true))

    corr_array = np.array(corrs, dtype=float)
    nan_mask = np.isnan(corr_array)
    n_nan = int(nan_mask.sum())

    if n_nan > 0:
        message = (
            f"{n_nan} of {len(corr_array)} time steps produced NaN Spearman "
            "correlation (constant predictions or fewer than 2 valid values). "
            "These rows are excluded."
        )
        if nan_policy == "raise":
            raise ValueError(message)
        elif nan_policy == "warn":
            warnings.warn(message, UserWarning, stacklevel=2)

    valid_corrs = corr_array[~nan_mask]

    if len(valid_corrs) < 2:
        warnings.warn(
            "Fewer than 2 valid time steps — cannot compute Spearman-Sharpe score. "
            "Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    mean_r = float(np.mean(valid_corrs))
    std_r = float(np.std(valid_corrs, ddof=ddof))

    if std_r == 0.0:
        warnings.warn(
            "Standard deviation of per-period Spearman correlations is zero. "
            "Score is undefined (all periods had identical correlation). "
            "Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    return mean_r / std_r


def per_period_spearman(
    y_pred: pd.DataFrame,
    y_true: pd.DataFrame,
) -> pd.Series:
    """Spearman correlation for each time step, returned as a Series.

    Useful for diagnosing model consistency and spotting regime-dependent
    performance. The Spearman-Sharpe score is mean / std of this series.

    Args:
        y_pred: DataFrame of shape (T, N).
        y_true: DataFrame of shape (T, N).

    Returns:
        pd.Series of length T with per-period Spearman correlations.
        Index matches y_pred.index. NaN where correlation is undefined.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}."
        )

    corrs = [
        spearman_corr_row(
            y_pred.iloc[t].to_numpy(dtype=float),
            y_true.iloc[t].to_numpy(dtype=float),
        )
        for t in range(len(y_pred))
    ]

    return pd.Series(corrs, index=y_pred.index, name="spearman_corr")


# ============================================================================
# Offline proxy metric — Pearson correlation (faster, not official)
# ============================================================================

def pearson_corr_row(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Pearson correlation between two 1-D arrays.

    Offline proxy only — the competition uses Spearman-based ranking.
    Use this for fast iteration and sanity checks; always validate
    final results with spearman_sharpe_score.

    Args:
        y_pred: Predicted values for a single time step, shape (n_assets,).
        y_true: Actual values for a single time step, shape (n_assets,).

    Returns:
        Pearson correlation in [-1, 1], or NaN if either input has zero
        variance or fewer than 2 valid pairs remain after NaN removal.

    Raises:
        ValueError: If y_pred and y_true have different lengths.
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

    if y_pred_clean.std() == 0.0 or y_true_clean.std() == 0.0:
        return float("nan")

    return float(np.corrcoef(y_pred_clean, y_true_clean)[0, 1])


def mean_row_pearson(
    y_pred: pd.DataFrame,
    y_true: pd.DataFrame,
    nan_policy: Literal["warn", "raise", "ignore"] = "warn",
) -> float:
    """Mean row-wise Pearson correlation — offline proxy metric.

    NOT the official competition metric. Use spearman_sharpe_score for
    final evaluation. Useful for fast iteration during development.

    Computes Pearson correlation row-by-row and returns the mean. Rows
    where correlation is NaN are excluded with an optional warning.

    Args:
        y_pred: DataFrame of shape (T, N).
        y_true: DataFrame of the same shape and column order.
        nan_policy: "warn" (default), "raise", or "ignore".

    Returns:
        Mean Pearson correlation across valid time steps. NaN if all rows
        produce NaN correlations.

    Raises:
        ValueError: If shapes or columns do not match.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred and y_true must have identical shapes. "
            f"Got {y_pred.shape} and {y_true.shape}."
        )
    if list(y_pred.columns) != list(y_true.columns):
        raise ValueError(
            "y_pred and y_true must have identical column names. "
            "Reorder or rename columns before calling this function."
        )

    corrs: list[float] = []
    for t in range(len(y_pred)):
        row_pred = y_pred.iloc[t].to_numpy(dtype=float)
        row_true = y_true.iloc[t].to_numpy(dtype=float)
        corrs.append(pearson_corr_row(row_pred, row_true))

    corr_array = np.array(corrs, dtype=float)
    nan_mask = np.isnan(corr_array)
    n_nan = int(nan_mask.sum())

    if n_nan > 0:
        message = (
            f"{n_nan} of {len(corr_array)} time steps produced NaN correlation "
            "(constant predictions or fewer than 2 valid values). "
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
    """Pearson correlation for each time step — offline proxy diagnostic.

    Args:
        y_pred: DataFrame of shape (T, N).
        y_true: DataFrame of shape (T, N).

    Returns:
        pd.Series of length T. Index matches y_pred.index.
        NaN where correlation is undefined.
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
