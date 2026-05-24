# Validation Strategy

## Why Standard K-Fold Is Wrong Here

Random K-Fold cross-validation splits data points randomly across folds. In time-series financial data, this causes **temporal leakage**: a model can train on future data points and validate on past ones, producing unrealistically optimistic estimates.

Example of leakage:
- Train set includes returns from 2022
- Validation set includes returns from 2020
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
- One unusual period (e.g., pandemic, market crash) can dominate the estimate
- No feedback on how the model generalizes across different regimes

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
- Mimics real-world deployment: model is re-trained as new data arrives

**Cons:**
- Early folds have very little training data → early estimates may be noisy
- Computationally expensive (one model fit per fold)
- Each fold's training set is a superset of the previous — folds are not independent

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
- Each fold uses the same amount of training data → more comparable estimates across folds
- Better captures non-stationarity: earlier data is discarded (prevents outdated patterns from dominating)

**Cons:**
- Throws away data when training window is limited
- Harder to tune window size
- Models that improve with more data may be underestimated

**When to use:** When there is evidence that market regime shifts make older data less useful.

---

### 4. Purged K-Fold with Embargo (Advanced)

```
Fold k: |─ TRAIN ─|─ GAP ─|─ VAL ─|─ GAP ─|
```

**How it works:** Same as walk-forward, but add a **purge period** (gap) between train and validation to prevent label leakage from overlapping feature windows. An **embargo** period after validation prevents test contamination.

**Pros:**
- Handles situations where features use long lookback windows (e.g., 60-day rolling mean)
- Standard in industrial financial ML (see López de Prado, *Advances in Financial Machine Learning*)

**Cons:**
- Requires careful parameterization (gap = max feature lookback)
- More complex to implement
- Reduces effective sample size

**When to use:** When feature engineering uses windows that overlap across the train/val boundary.

---

## Recommended Plan

| Stage | Strategy | Reason |
|-------|----------|--------|
| Stage 1-2 | Holdout (last 20% of time) | Speed; baseline comparisons |
| Stage 3 | Walk-forward (expanding) with 5 folds | Robust model selection |
| Stage 4 | Walk-forward with embargo | Final model, if features use long lookbacks |

---

## Gap Size Calculation

If your features use a maximum lookback window of $L$ periods, set the purge gap to $L$ periods between train and validation:

```python
gap = max_feature_lookback_periods  # e.g., 60 for a 60-day rolling mean
```

---

## Metric During Validation

Use the same metric as the competition: **mean Pearson correlation across time steps**. Do not optimize or report RMSE or MAE as your primary validation signal.

```python
from src.evaluation.metric import mean_row_pearson
score = mean_row_pearson(y_pred_df, y_true_df)
```

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. [Chapter 7: Cross-Validation in Finance]
- Bailey, D., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management*.
