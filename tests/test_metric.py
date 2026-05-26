"""
tests/test_metric.py
--------------------
Unit tests for src/evaluation/metric.py.

Covers both the official competition metric (Spearman-Sharpe) and the
offline Pearson proxy. Each test uses small, hand-verifiable synthetic
examples so expected values can be checked without running the full dataset.

Run with:
    pytest tests/test_metric.py -v
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metric import (
    mean_row_pearson,
    pearson_corr_row,
    per_period_pearson,
    per_period_spearman,
    spearman_corr_row,
    spearman_sharpe_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(*rows: tuple[float, ...], columns: list[str] | None = None) -> pd.DataFrame:
    """Build a DataFrame from row tuples."""
    data = list(rows)
    cols = columns or [str(i) for i in range(len(data[0]))]
    return pd.DataFrame(data, columns=cols)


# ---------------------------------------------------------------------------
# pearson_corr_row (offline proxy)
# ---------------------------------------------------------------------------

class TestPearsonCorrRow:
    def test_perfect_positive_correlation(self):
        """Identical rankings → correlation = 1.0."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([10.0, 20.0, 30.0])
        assert math.isclose(pearson_corr_row(pred, true), 1.0, abs_tol=1e-9)

    def test_perfect_negative_correlation(self):
        """Reversed rankings → correlation = -1.0."""
        pred = np.array([3.0, 2.0, 1.0])
        true = np.array([1.0, 2.0, 3.0])
        assert math.isclose(pearson_corr_row(pred, true), -1.0, abs_tol=1e-9)

    def test_zero_correlation(self):
        """Orthogonal vectors → correlation near 0."""
        pred = np.array([1.0, -1.0, 1.0, -1.0])
        true = np.array([1.0, 1.0, -1.0, -1.0])
        result = pearson_corr_row(pred, true)
        assert math.isclose(result, 0.0, abs_tol=1e-9)

    def test_scale_invariance(self):
        """Multiplying predictions by a constant does not change correlation."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.5, 2.5, 3.5])
        base = pearson_corr_row(pred, true)
        scaled = pearson_corr_row(pred * 100, true)
        assert math.isclose(base, scaled, abs_tol=1e-9)

    def test_mean_shift_invariance(self):
        """Adding a constant to predictions does not change correlation."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.5, 2.5, 3.5])
        base = pearson_corr_row(pred, true)
        shifted = pearson_corr_row(pred + 999, true)
        assert math.isclose(base, shifted, abs_tol=1e-9)

    def test_constant_predictions_return_nan(self):
        """Constant predictions have zero variance → NaN."""
        pred = np.array([0.5, 0.5, 0.5])
        true = np.array([1.0, 2.0, 3.0])
        result = pearson_corr_row(pred, true)
        assert math.isnan(result)

    def test_constant_targets_return_nan(self):
        """Constant targets have zero variance → NaN."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([0.0, 0.0, 0.0])
        result = pearson_corr_row(pred, true)
        assert math.isnan(result)

    def test_nan_values_excluded_pairwise(self):
        """NaN values in pred or true are excluded pairwise."""
        pred = np.array([1.0, np.nan, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        result = pearson_corr_row(pred, true)
        expected = pearson_corr_row(np.array([1.0, 3.0]), np.array([1.0, 3.0]))
        assert math.isclose(result, expected, abs_tol=1e-9)

    def test_fewer_than_two_valid_values_returns_nan(self):
        """Only one valid pair → NaN."""
        pred = np.array([1.0, np.nan, np.nan])
        true = np.array([1.0, np.nan, np.nan])
        result = pearson_corr_row(pred, true)
        assert math.isnan(result)

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            pearson_corr_row(np.array([1.0, 2.0]), np.array([1.0]))

    def test_two_assets(self):
        """Works correctly with exactly two assets."""
        pred = np.array([1.0, 2.0])
        true = np.array([1.0, 2.0])
        result = pearson_corr_row(pred, true)
        assert math.isclose(result, 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# mean_row_pearson (offline proxy)
# ---------------------------------------------------------------------------

class TestMeanRowPearson:
    def test_perfect_positive_all_periods(self):
        """All periods perfectly correlated → score = 1.0."""
        pred = make_df((1, 2, 3), (4, 5, 6), (7, 8, 9))
        true = make_df((10, 20, 30), (40, 50, 60), (70, 80, 90))
        assert math.isclose(mean_row_pearson(pred, true), 1.0, abs_tol=1e-9)

    def test_perfect_negative_all_periods(self):
        """All periods perfectly anti-correlated → score = -1.0."""
        pred = make_df((3, 2, 1), (6, 5, 4))
        true = make_df((1, 2, 3), (4, 5, 6))
        assert math.isclose(mean_row_pearson(pred, true), -1.0, abs_tol=1e-9)

    def test_mixed_periods_averages_correctly(self):
        """Mean of [1.0, -1.0] should be 0.0."""
        pred = make_df((1, 2, 3), (3, 2, 1))
        true = make_df((1, 2, 3), (1, 2, 3))
        result = mean_row_pearson(pred, true, nan_policy="ignore")
        assert math.isclose(result, 0.0, abs_tol=1e-9)

    def test_shape_mismatch_raises(self):
        """Different shapes raise ValueError."""
        pred = make_df((1, 2, 3), (4, 5, 6))
        true = make_df((1, 2, 3))
        with pytest.raises(ValueError, match="shape"):
            mean_row_pearson(pred, true)

    def test_column_mismatch_raises(self):
        """Different column names raise ValueError."""
        pred = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        true = pd.DataFrame({"A": [1, 2], "C": [3, 4]})
        with pytest.raises(ValueError, match="column names"):
            mean_row_pearson(pred, true)

    def test_nan_row_excluded_with_warning(self):
        """Rows producing NaN correlation are excluded, with a warning."""
        pred = make_df((1, 2, 3), (1, 1, 1))  # second row: constant
        true = make_df((1, 2, 3), (1, 2, 3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = mean_row_pearson(pred, true, nan_policy="warn")
        assert any("NaN" in str(warning.message) for warning in w)
        assert math.isclose(result, 1.0, abs_tol=1e-9)

    def test_nan_policy_raise(self):
        """nan_policy='raise' raises ValueError on NaN correlation."""
        pred = make_df((1, 2, 3), (1, 1, 1))
        true = make_df((1, 2, 3), (1, 2, 3))
        with pytest.raises(ValueError, match="NaN"):
            mean_row_pearson(pred, true, nan_policy="raise")

    def test_nan_policy_ignore(self):
        """nan_policy='ignore' silently excludes NaN rows."""
        pred = make_df((1, 2, 3), (1, 1, 1))
        true = make_df((1, 2, 3), (1, 2, 3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = mean_row_pearson(pred, true, nan_policy="ignore")
        assert not any("NaN" in str(warning.message) for warning in w)
        assert math.isclose(result, 1.0, abs_tol=1e-9)

    def test_all_nan_returns_nan(self):
        """All rows producing NaN → overall NaN."""
        pred = make_df((1, 1, 1), (2, 2, 2))  # both rows: constant
        true = make_df((1, 2, 3), (4, 5, 6))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mean_row_pearson(pred, true, nan_policy="ignore")
        assert math.isnan(result)

    def test_single_row(self):
        """Works with a single time step."""
        pred = make_df((1, 2, 3))
        true = make_df((1, 2, 3))
        assert math.isclose(mean_row_pearson(pred, true), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# per_period_pearson (offline proxy)
# ---------------------------------------------------------------------------

class TestPerPeriodPearson:
    def test_returns_series_of_correct_length(self):
        pred = make_df((1, 2, 3), (4, 5, 6), (7, 8, 9))
        true = make_df((1, 2, 3), (4, 5, 6), (7, 8, 9))
        result = per_period_pearson(pred, true)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_series_index_matches_input(self):
        """Series index should match the DataFrame's index."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        pred = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]}, index=idx)
        true = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]}, index=idx)
        result = per_period_pearson(pred, true)
        assert list(result.index) == list(idx)

    def test_values_are_correct(self):
        """Check per-period values against expected correlations."""
        pred = make_df((1, 2, 3), (3, 2, 1))
        true = make_df((1, 2, 3), (1, 2, 3))
        result = per_period_pearson(pred, true)
        assert math.isclose(result.iloc[0], 1.0, abs_tol=1e-9)
        assert math.isclose(result.iloc[1], -1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# spearman_corr_row (official metric component)
# ---------------------------------------------------------------------------

class TestSpearmanCorrRow:
    def test_perfect_positive_correlation(self):
        """Identical rankings → Spearman = 1.0."""
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([10.0, 20.0, 30.0, 40.0])
        assert math.isclose(spearman_corr_row(pred, true), 1.0, abs_tol=1e-9)

    def test_perfect_negative_correlation(self):
        """Reversed rankings → Spearman = -1.0."""
        pred = np.array([4.0, 3.0, 2.0, 1.0])
        true = np.array([1.0, 2.0, 3.0, 4.0])
        assert math.isclose(spearman_corr_row(pred, true), -1.0, abs_tol=1e-9)

    def test_rank_invariance(self):
        """Spearman is invariant to monotone transforms of predictions."""
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([1.5, 2.5, 3.5, 4.5])
        base = spearman_corr_row(pred, true)
        # Apply a nonlinear monotone transform — rank order preserved
        transformed = spearman_corr_row(pred ** 3, true)
        assert math.isclose(base, transformed, abs_tol=1e-9)

    def test_spearman_differs_from_pearson_with_outlier(self):
        """Spearman and Pearson diverge when outliers are present.

        pred has rank order [1, 2, 3, 4] matching true → Spearman = 1.0.
        But the outlier magnitude (100 vs 4) breaks linearity → Pearson < 1.0.
        """
        pred = np.array([1.0, 2.0, 3.0, 100.0])  # rank order matches true exactly
        true = np.array([1.0, 2.0, 3.0, 4.0])
        s = spearman_corr_row(pred, true)
        p = pearson_corr_row(pred, true)
        # Spearman sees perfect rank match → 1.0
        assert math.isclose(s, 1.0, abs_tol=1e-9)
        # Pearson is distorted by outlier magnitude → strictly less than 1.0
        assert p < 1.0
        assert p > 0  # still positive (rank order preserved)

        # With a rank-reversed outlier: Spearman captures the reversal
        pred2 = np.array([1.0, 2.0, 3.0, -100.0])  # -100 is now ranked lowest
        s2 = spearman_corr_row(pred2, true)  # rank [3,2,1,4] vs [1,2,3,4] -- not perfect
        p2 = pearson_corr_row(pred2, true)
        assert s2 < 1.0  # Spearman < 1 because -100 disrupts rank match
        assert p2 < 0    # Pearson negative due to large outlier pulling correlation negative

    def test_nan_excluded_pairwise(self):
        """NaN values are excluded pairwise before rank computation."""
        pred = np.array([1.0, np.nan, 3.0, 4.0])
        true = np.array([1.0, 2.0, 3.0, 4.0])
        result = spearman_corr_row(pred, true)
        expected = spearman_corr_row(
            np.array([1.0, 3.0, 4.0]), np.array([1.0, 3.0, 4.0])
        )
        assert math.isclose(result, expected, abs_tol=1e-9)

    def test_fewer_than_two_valid_returns_nan(self):
        """Fewer than 2 valid pairs → NaN."""
        pred = np.array([1.0, np.nan, np.nan])
        true = np.array([1.0, 2.0, 3.0])
        assert math.isnan(spearman_corr_row(pred, true))

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            spearman_corr_row(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# spearman_sharpe_score (official competition metric)
# ---------------------------------------------------------------------------

class TestSpearmanSharpeScore:
    def test_consistent_perfect_correlation(self):
        """All periods Spearman=1.0 → std=0 → score=NaN (undefined)."""
        pred = make_df((1, 2, 3), (4, 5, 6), (7, 8, 9))
        true = make_df((1, 2, 3), (4, 5, 6), (7, 8, 9))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = spearman_sharpe_score(pred, true)
        assert math.isnan(result)

    def test_mixed_positive_negative(self):
        """Mean = 0, std > 0 → score = 0."""
        pred = make_df((1, 2, 3), (3, 2, 1))
        true = make_df((1, 2, 3), (1, 2, 3))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = spearman_sharpe_score(pred, true, nan_policy="ignore")
        assert math.isclose(result, 0.0, abs_tol=1e-9)

    def test_positive_mean_positive_std_gives_positive_score(self):
        """Mostly positive correlations → positive Sharpe score."""
        pred = make_df((1, 2, 3), (1, 2, 3), (1, 2, 3), (3, 2, 1))
        true = make_df((1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = spearman_sharpe_score(pred, true, nan_policy="ignore")
        assert result > 0

    def test_shape_mismatch_raises(self):
        """Different shapes raise ValueError."""
        pred = make_df((1, 2, 3), (4, 5, 6))
        true = make_df((1, 2, 3))
        with pytest.raises(ValueError, match="shape"):
            spearman_sharpe_score(pred, true)

    def test_column_mismatch_raises(self):
        """Different column names raise ValueError."""
        pred = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        true = pd.DataFrame({"A": [1, 2], "C": [3, 4]})
        with pytest.raises(ValueError, match="column names"):
            spearman_sharpe_score(pred, true)

    def test_single_valid_period_returns_nan(self):
        """Only one valid time step — cannot compute std → NaN."""
        pred = make_df((1, 2, 3), (1, 1, 1))  # second row is constant → NaN corr
        true = make_df((1, 2, 3), (1, 2, 3))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = spearman_sharpe_score(pred, true)
        assert math.isnan(result)

    def test_nan_policy_raise(self):
        """nan_policy='raise' raises ValueError on NaN Spearman rows."""
        pred = make_df((1, 2, 3), (1, 1, 1))
        true = make_df((1, 2, 3), (1, 2, 3))
        with pytest.raises(ValueError, match="NaN"):
            spearman_sharpe_score(pred, true, nan_policy="raise")

    def test_score_formula_matches_manual_calculation(self):
        """Verify score = mean/std matches manual calculation."""
        rng = np.random.default_rng(0)
        T, N = 20, 10
        pred = pd.DataFrame(rng.standard_normal((T, N)))
        true = pd.DataFrame(rng.standard_normal((T, N)))

        per_period = per_period_spearman(pred, true)
        valid = per_period.dropna()
        expected = float(valid.mean() / valid.std(ddof=1))

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = spearman_sharpe_score(pred, true, nan_policy="ignore")

        assert math.isclose(result, expected, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# per_period_spearman (official metric diagnostic)
# ---------------------------------------------------------------------------

class TestPerPeriodSpearman:
    def test_returns_series_of_correct_length(self):
        pred = make_df((1, 2, 3), (4, 5, 6), (7, 8, 9))
        true = make_df((1, 2, 3), (4, 5, 6), (7, 8, 9))
        result = per_period_spearman(pred, true)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_series_name_is_spearman_corr(self):
        pred = make_df((1, 2, 3))
        true = make_df((1, 2, 3))
        result = per_period_spearman(pred, true)
        assert result.name == "spearman_corr"

    def test_perfect_positive_each_period(self):
        """All periods Spearman=1.0."""
        pred = make_df((1, 2, 3), (4, 5, 6))
        true = make_df((10, 20, 30), (40, 50, 60))
        result = per_period_spearman(pred, true)
        assert all(math.isclose(v, 1.0, abs_tol=1e-9) for v in result)

    def test_index_matches_input(self):
        """Series index should match the DataFrame's index."""
        idx = [10, 20, 30]
        pred = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]}, index=idx)
        true = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]}, index=idx)
        result = per_period_spearman(pred, true)
        assert list(result.index) == idx

    def test_shape_mismatch_raises(self):
        """Different shapes raise ValueError."""
        pred = make_df((1, 2, 3))
        true = make_df((1, 2, 3), (4, 5, 6))
        with pytest.raises(ValueError, match="Shape mismatch"):
            per_period_spearman(pred, true)
