# Validation Strategy

## Why Standard K-Fold Is Wrong Here

Random K-Fold cross-validation splits data points randomly across folds. In time-series financial data, this causes **temporal leakage**: a model can train on future data points and validate on past ones, producing unrealistically optimistic estimates.

Example of leakage:
- Train set includes returns from date_id 1500+
- Validation set includes returns from date_id 500
- The model learns patterns from the "future" that it will never have at prediction time

**Always use temporal splits.** Every validation scheme below respects the arrow of time.

---

## Candidate Validation Strategies

### 1. Simple Train / Validation Split (Holdout)

```
|─── TRAIN ──────────────────|── VALIDATION ──|
t=0                        t=T-k            t=T
```

**How it works:** Use the most recent `k` time periods as validation. Train on everything before.

**Pros:**
- Simple to implement
- Matches competition test structure closely if test set is the most recent period
- Fast (one fit)

**Cons:**
- High variance — validation score depends heavily on which period you hold out
- One unusual period (e.g., sharp macro event) can dominate the estimate
- No feedback on how the model generalises across different regimes

**When to use:** Quick sanity checks, early experimentation.

---

### 2. Walk-Forward Validation (Expanding Window)

```
Fold 1: |─ TRAIN ─|─ VAL ─|
Fold 2: |── TRAIN ──|─ VAL ─|
Fold 3: |─── TRAIN ───|─ VAL ─|
Fold 4: |──── TRAIN ────|─ VAL ─|
```

**How it works:** Train on all data up to time $t$; validate on the next $k$ periods. Slide $t$ forward and repeat.

**Pros:**
- Tests how the model performs as it sees more data (growing training set)
- Multiple validation windows → lower variance estimate
- Mimics real-world deployment

**Cons:**
- Early folds have very little training data → early estimates may be noisy
- Computationally expensive (one model fit per fold)

**When to use:** Primary validation strategy for most experiments.

---

### 3. Walk-Forward Validation (Rolling Window)

```
Fold 1: |─ TRAIN ─|─ VAL ─|
Fold 2:   |─ TRAIN ─|─ VAL ─|
Fold 3:     |─ TRAIN ─|─ VAL ─|
```

**How it works:** Fixed-size training window slides forward in time.

**Pros:**
- Each fold uses the same amount of training data → more comparable estimates
- Better captures non-stationarity: stale data is discarded

**Cons:**
- Throws away data when training window is limited
- Harder to tune window size

**When to use:** When there is evidence that market regime shifts make older data less useful.

---

### 4. Purged Walk-Forward with Embargo (Advanced)

```
Fold k: |─ TRAIN ─|─ GAP ─|─ VAL ─|
```

**How it works:** Add a **purge gap** between train and validation to prevent feature-window overlap. If features use a 20-day rolling mean, a 20-day gap ensures no training sample contributes features computed from validation period data.

**Pros:**
- Correct when features use long lookback windows
- Standard in financial ML (López de Prado, *Advances in Financial Machine Learning*)

**Cons:**
- Reduces effective sample size
- Requires careful gap parameterisation (gap = max feature lookback in days)

**When to use:** Once feature engineering introduces rolling windows longer than a few days.

---

## Recommended Plan

| Stage | Strategy | Reason |
|-------|----------|--------|
| Stage 2 | Holdout (last 20% of date_id range) | Speed; baseline comparisons |
| Stage 3 | Walk-forward expanding, 5 folds | Robust model selection |
| Stage 4 | Walk-forward with purge gap | Final model, if features use long lookbacks |

For Stage 2 holdout:
- Training: `date_id` 0–1568 (~80% of 1961 rows)
- Validation: `date_id` 1569–1960 (~392 rows, ~6 months of trading days)

---

## Gap Size Calculation

If features use a maximum lookback window of $L$ periods, set the purge gap to $L$:

```python
gap_periods = max_feature_lookback  # e.g. 20 for a 20-day rolling mean
```

With date_id as an integer counter, period boundaries are simple arithmetic:

```python
train_end   = split_date_id - gap_periods
val_start   = split_date_id
```

---

## Metric During Validation

Use the **official competition metric** during validation:

```python
from src.evaluation.metric import spearman_sharpe_score

score = spearman_sharpe_score(y_pred_df, y_true_df)
```

The Pearson proxy (`mean_row_pearson`) may also be logged for reference, but do not
use it as the primary model selection signal.

Reporting template for each experiment:

```
Fold: <fold_id>   Train: <date_id_range>   Val: <date_id_range>
Spearman-Sharpe : <score>
Mean Spearman   : <mean>   Std Spearman: <std>
Pearson proxy   : <score>  (for reference only)
```

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. [Chapter 7: Cross-Validation in Finance]
- Bailey, D., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management*.
