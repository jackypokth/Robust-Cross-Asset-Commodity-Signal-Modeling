# Metric Explainer

## Official Competition Metric: Spearman-Sharpe Score

The official competition metric is a **Sharpe-like ratio of Spearman rank correlations**:

$$\text{Score} = \frac{\overline{r^{\rho}}}{\sigma(r^{\rho})}$$

where $r^{\rho}_t$ is the **Spearman rank correlation** between predicted and actual
target values across all 424 targets at time step $t$, and the mean and standard
deviation are taken across all valid time steps.

---

## Step-by-Step Breakdown

**Step 1 — Per time step, compute Spearman correlation.**

At each time step $t$, rank all 424 predicted target values and all 424 actual target
values separately. Compute the Pearson correlation of those two rank vectors:

$$r^{\rho}_t = \text{Spearman}(\hat{y}_t, y_t)$$

Spearman correlation is $+1$ if your predicted ranking perfectly matches actual; $-1$
if perfectly inverted; $0$ if uncorrelated.

**Step 2 — Average the per-period correlations.**

$$\overline{r^{\rho}} = \frac{1}{T} \sum_{t=1}^{T} r^{\rho}_t$$

**Step 3 — Divide by the standard deviation of per-period correlations.**

$$\text{Score} = \frac{\overline{r^{\rho}}}{\text{std}(r^{\rho}_1, \ldots, r^{\rho}_T)}$$

This is the final leaderboard score.

---

## Why Divide by Standard Deviation?

The denominator penalises **inconsistency**. A model with stable, moderate performance
scores higher than one with occasional great predictions mixed with bad ones.

| Model | Mean $r^\rho$ | Std $r^\rho$ | Score |
|-------|--------------|-------------|-------|
| Stable moderate | 0.30 | 0.10 | **3.00** |
| Volatile | 0.30 | 0.60 | 0.50 |
| Consistently bad | -0.10 | 0.05 | -2.00 |

This mirrors the Sharpe ratio in portfolio theory: raw returns adjusted for volatility.

---

## Why Spearman, Not Pearson?

**Spearman** is rank-based — it is invariant to any monotone transformation of the
predictions. Predicting `[0.01, 0.02, 0.03]` or `[1, 4, 9]` gives the same Spearman
score if the ranking is the same.

**Pearson** is more sensitive to the scale and distribution of prediction magnitudes.
In the presence of outlier targets (common in commodity spreads), Pearson can be
distorted by a few extreme values. Spearman is more robust.

For practical development, Pearson and Spearman often agree closely, which is why the
offline proxy (Pearson) is still useful for quick iteration.

---

## Key Properties of the Official Metric

| Property | Implication |
|----------|-------------|
| Rank-based (Spearman) | Scale and distribution of predictions don't matter; only ranking does |
| Sharpe-like denominator | Consistent models beat volatile ones with the same mean |
| Computed per row (time step) | One bad period hurts your Sharpe ratio twice: bad mean AND high std |
| NaN handling | Time steps with constant predictions are excluded |
| Sensitive to consistency | Outlier predictions in single periods can collapse your score |

---

## Offline Proxy: Mean Row-wise Pearson

The repo also implements **mean row-wise Pearson correlation** (`mean_row_pearson` in
`src/evaluation/metric.py`). This is:

$$\text{Proxy Score} = \frac{1}{T} \sum_{t=1}^{T} \text{PearsonCorr}(\hat{y}_t, y_t)$$

**When to use the proxy:**
- Quick sanity checks during feature engineering.
- Verifying that predictions are not degenerate (constant, NaN).
- Early-stage development where you want a fast signal.

**When NOT to use the proxy as your final signal:**
- Ranking models or comparing experiments.
- Reporting results.
- Anything that goes in a commit message or notebook conclusion.

Always validate final results with `spearman_sharpe_score`.

---

## Local Metric Reference

```python
from src.evaluation.metric import spearman_sharpe_score, per_period_spearman
from src.evaluation.metric import mean_row_pearson  # offline proxy only

# Official metric
score = spearman_sharpe_score(y_pred_df, y_true_df)

# Diagnostic: per-period breakdown
per_period = per_period_spearman(y_pred_df, y_true_df)
print(per_period.describe())

# Offline proxy (fast iteration)
proxy = mean_row_pearson(y_pred_df, y_true_df)
```

See `tests/test_metric.py` for examples with known expected values.

---

## Common Pitfalls

1. **Optimizing MSE instead of correlation**: MSE-minimizing models do not maximise
   cross-sectional ranking. Train with a ranking-aware loss or use correlation directly.

2. **Ignoring the Sharpe denominator**: A model that produces high but volatile
   correlation (e.g. great in trending markets, terrible in ranging markets) will score
   poorly. Stability matters as much as mean performance.

3. **Using Pearson as the final metric**: Pearson and Spearman can diverge significantly
   when outlier targets are present. Always confirm with `spearman_sharpe_score`.

4. **Optimising on all 424 targets jointly without considering lags**: Targets at
   different lags (1, 2, 3, 4) have different noise levels and autocorrelation
   structures. Mixed-lag evaluation may hide lag-specific weaknesses.
