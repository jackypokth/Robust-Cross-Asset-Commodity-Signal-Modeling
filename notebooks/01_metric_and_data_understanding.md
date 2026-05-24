# Notebook 01: Metric and Data Understanding

**Status:** Notebook plan — convert to `.ipynb` after competition data is downloaded.

This document describes the cell-by-cell structure of the first exploratory notebook. The actual `.ipynb` file should be created once `data/raw/` contains the competition data, since cells that load and display data require real inputs to be meaningful.

---

## Purpose

1. Verify the local metric implementation against sanity-check examples.
2. Load and inspect the raw data files.
3. Build an initial understanding of the return distribution, missingness, and temporal structure.
4. Establish a baseline score (naive model) to beat in Stage 2.

---

## Cell Plan

### Cell 1 — Imports and Setup

```python
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.paths import check_raw_data, ensure_dirs
from src.data.load_data import load_train, load_test, pivot_to_wide
from src.evaluation.metric import mean_row_pearson, per_period_pearson

# Suppress non-critical warnings for clean notebook output
warnings.filterwarnings("ignore", category=FutureWarning)

# Check data availability
print(check_raw_data())
```

**Expected output:** Dict showing which raw data files are present.

---

### Cell 2 — Verify Metric Implementation

Run metric on synthetic examples to confirm it behaves as expected before touching real data.

```python
# Perfect positive correlation
pred_perfect = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
true_perfect = pd.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]})
assert abs(mean_row_pearson(pred_perfect, true_perfect) - 1.0) < 1e-9
print("Perfect correlation test: PASSED")

# Perfect negative correlation
pred_neg = pd.DataFrame({"A": [3, 2], "B": [1, 0]})
true_neg = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
assert abs(mean_row_pearson(pred_neg, true_neg) - (-1.0)) < 1e-9
print("Negative correlation test: PASSED")

# Naive zero-prediction baseline
n_times, n_assets = 100, 10
rng = np.random.default_rng(42)
true_random = pd.DataFrame(rng.standard_normal((n_times, n_assets)))
pred_zeros  = pd.DataFrame(np.zeros((n_times, n_assets)))
score_zeros = mean_row_pearson(pred_zeros, true_random, nan_policy="ignore")
print(f"Zero-prediction baseline score: {score_zeros:.4f}  (expect ≈ 0.0)")
```

---

### Cell 3 — Load Training Data

```python
train = load_train()
print(f"Shape: {train.shape}")
print(f"Columns: {list(train.columns)}")
print(f"Date range: {train['date'].min()} to {train['date'].max()}")
print(f"Unique commodities: {train['commodity'].nunique()}")
print()
train.head()
```

---

### Cell 4 — Missing Data Analysis

```python
missing = train.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0].sort_values(ascending=False))

# Heatmap of missingness over time (if not too large)
# pivot = train.pivot(index='date', columns='commodity', values='target')
# sns.heatmap(pivot.isnull(), cbar=False, yticklabels=False)
# plt.title("Missing target values by date and commodity")
# plt.tight_layout(); plt.show()
```

---

### Cell 5 — Return Distribution

```python
# Distribution of target values per commodity
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Overall
train["target"].hist(bins=100, ax=axes[0])
axes[0].set_title("Overall target distribution")
axes[0].set_xlabel("Return")
axes[0].set_ylabel("Count")

# Per commodity (boxplot)
train.boxplot(column="target", by="commodity", ax=axes[1], rot=45)
axes[1].set_title("Target distribution by commodity")
axes[1].set_xlabel("")
plt.suptitle("")
plt.tight_layout(); plt.show()

# Summary statistics
print(train.groupby("commodity")["target"].describe().round(4))
```

---

### Cell 6 — Temporal Structure

```python
# Number of observations per time period
obs_per_date = train.groupby("date")["target"].count()
obs_per_date.plot(figsize=(12, 3), title="Observations per time step")
plt.ylabel("Count"); plt.tight_layout(); plt.show()

# Check if all commodities appear at all time steps
pivot = pivot_to_wide(train)
print(f"Wide shape: {pivot.shape}  (dates × commodities)")
print(f"NaN fraction: {pivot.isnull().mean().mean():.2%}")
```

---

### Cell 7 — Autocorrelation

```python
# Autocorrelation of target returns per commodity
from pandas.plotting import autocorrelation_plot

sample_commodity = train["commodity"].unique()[0]
returns = train[train["commodity"] == sample_commodity]["target"].dropna()

fig, ax = plt.subplots(figsize=(10, 3))
autocorrelation_plot(returns, ax=ax)
ax.set_title(f"Autocorrelation of target — {sample_commodity}")
ax.set_xlim(0, 30)
plt.tight_layout(); plt.show()
```

---

### Cell 8 — Cross-Commodity Correlation

```python
# Pairwise correlation matrix of returns across commodities
corr_matrix = pivot.corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
)
plt.title("Cross-commodity return correlation")
plt.tight_layout(); plt.show()
```

---

### Cell 9 — Naive Baseline Score

```python
# Naive baseline 1: predict zero for all commodities at all times
wide_true = pivot_to_wide(train)
wide_zero = pd.DataFrame(0.0, index=wide_true.index, columns=wide_true.columns)
score_zero = mean_row_pearson(wide_zero, wide_true, nan_policy="ignore")
print(f"Zero-prediction baseline: {score_zero:.4f}")

# Naive baseline 2: predict last period's return (momentum)
wide_last = wide_true.shift(1)
score_last = mean_row_pearson(wide_last.iloc[1:], wide_true.iloc[1:], nan_policy="ignore")
print(f"Last-return (momentum) baseline: {score_last:.4f}")

# Naive baseline 3: predict cross-sectional mean (no variation)
wide_mean = wide_true.mean(axis=1).to_frame().dot(
    pd.DataFrame([[1] * wide_true.shape[1]], columns=wide_true.columns)
)
score_mean = mean_row_pearson(wide_mean, wide_true, nan_policy="ignore")
print(f"Cross-sectional mean baseline: {score_mean:.4f}")
```

---

### Cell 10 — Key Takeaways

Fill in after running the notebook with real data:

```markdown
## Key observations from Cell 10:

- Dataset spans: TBD
- Number of commodities: TBD
- Average missingness: TBD%
- Zero-prediction baseline score: TBD
- Momentum (lag-1) baseline score: TBD
- Notable return outliers: TBD
- Commodities with highest cross-correlation: TBD
- Primary open questions for Stage 2: TBD
```

---

## How to Run

```bash
# Ensure data is downloaded
ls data/raw/

# Launch Jupyter
jupyter notebook notebooks/01_metric_and_data_understanding.ipynb

# Or run tests first to verify metric
pytest tests/test_metric.py -v
```
