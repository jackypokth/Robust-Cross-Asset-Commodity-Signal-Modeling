# Metric Explainer: Pearson Correlation (Row-wise Average)

## Plain English

The competition scores your predictions by measuring how well your **ranking of commodities** matches the actual ranking of commodity returns at each point in time.

At every time step in the test set:
1. Look at your predicted return for each commodity.
2. Look at the actual realized return for each commodity.
3. Compute the Pearson correlation between your predictions and the actual returns across commodities.
4. Repeat for every time step.
5. **Average** all those per-period correlations to get your final score.

A score of **1.0** means you perfectly ranked all commodities in every period.
A score of **0.0** means your predictions are no better than random.
A score of **-1.0** means you perfectly inverted every ranking (systematically wrong in a useful way).

---

## Mathematical Definition

Let $T$ be the number of time steps and $N$ be the number of commodities.

For each time step $t$:

$$r_t = \text{PearsonCorr}(\hat{y}_t, y_t)$$

where:
- $\hat{y}_t \in \mathbb{R}^N$ — your predicted returns for all commodities at time $t$
- $y_t \in \mathbb{R}^N$ — the actual realized returns for all commodities at time $t$

The Pearson correlation is:

$$r_t = \frac{\sum_{i=1}^{N}(\hat{y}_{t,i} - \bar{\hat{y}}_t)(y_{t,i} - \bar{y}_t)}{\sqrt{\sum_{i=1}^{N}(\hat{y}_{t,i} - \bar{\hat{y}}_t)^2} \cdot \sqrt{\sum_{i=1}^{N}(y_{t,i} - \bar{y}_t)^2}}$$

The final score is:

$$\text{Score} = \frac{1}{T} \sum_{t=1}^{T} r_t$$

---

## Key Properties

| Property | Implication |
|----------|-------------|
| Scale-invariant | Predicting 0.01 or 100 for every commodity makes no difference if the ranking is the same |
| Mean-shift-invariant | Adding a constant to all predictions doesn't change the score |
| Computed per row (time step) | One very bad prediction period cannot be offset by many mediocre ones — it hurts the average |
| NaN handling | If a time step has no valid predictions, it's typically excluded from the average |
| Sensitive to outliers | Pearson assumes roughly linear relationship; extreme return outliers can distort the correlation |

---

## Intuition: Correlation vs. MSE

**MSE** penalizes prediction magnitude errors. Predicting +5% when actual is +3% is penalized.

**Pearson correlation** only penalizes *ranking* errors. Predicting +2% (best) when actual return is +3% (best) is fine — you got the rank right, which is all that matters.

This means:
- You don't need to predict accurate return magnitudes.
- You need to correctly identify which commodities will outperform vs. underperform *at each point in time*.
- A model that predicts the right direction systematically — even with poor magnitude — scores well.

---

## Common Pitfalls

1. **Optimizing MSE locally instead of correlation**: A model that minimizes squared error may not maximize correlation.
2. **Treating this as a pure regression problem**: If you use MSE as your training loss without thinking about rankings, you may train a model that scores poorly on this metric.
3. **Forgetting that the metric is per time step**: A model that averages predictions across time is effectively removing the time dimension, which destroys the cross-sectional signal.

---

## Local Metric Reference

See `src/evaluation/metric.py` for the implementation.
See `tests/test_metric.py` for validation tests.

---

## Useful Approximation

Because the metric is correlation (not RMSE), you can think of this as a **cross-sectional ranking problem**:

> At each time $t$, rank all commodities. Predict scores that produce the same ranking.

Any monotone transform of your predictions preserves the Pearson score:
- Log, sigmoid, standardization — all fine
- Winsorization may help with outlier stability
