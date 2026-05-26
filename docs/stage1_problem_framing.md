# Stage 1: Problem Framing

## Objective

Establish a clear understanding of the competition before writing any modeling
code. This document records:

1. What is being predicted and how success is measured.
2. What structure the data likely has (before downloading).
3. What the modeling implications are.
4. Falsifiable baseline hypotheses to test in Stage 2.
5. Open research questions to answer during EDA.

---

## 1. What Is Being Predicted?

The competition asks for **future returns of commodity assets** at each time step.
A "return" is the proportional price change from one period to the next.

Key framing:
- We are not predicting raw prices — we predict **returns** (or signals that rank
  returns correctly).
- The evaluation metric (Pearson correlation) is **scale- and mean-shift-invariant**
  within each time step. Only the cross-sectional ordering of predicted returns matters.
- Predictions cover multiple commodities simultaneously, giving the problem a
  **cross-sectional dimension** on top of the time dimension.

---

## 2. Inferred Data Structure

Based on the competition description, the dataset is expected to be a **panel**
(long-format time series with multiple assets per date):

```
date | commodity_id | feature_1 | feature_2 | ... | target
```

- `date` — observation timestamp; frequency TBD (daily, weekly, or monthly).
- `commodity_id` — asset identifier.
- `feature_*` — lagged or engineered input features.
- `target` — future return to predict; likely log-return or % change.

> All column names above are **inferred** and must be verified after downloading
> the competition CSVs. See `docs/data_dictionary.md`.

---

## 3. Modeling Implications

| Observation | Implication |
|-------------|-------------|
| Pearson metric, not MSE | Scale is irrelevant; correct *ranking* of assets is what matters |
| Panel structure | Cross-sectional models can exploit relative patterns across assets |
| Financial time series | Non-stationarity, regime shifts, and autocorrelation must be accounted for |
| Ranking-aware objective | Linear models, tree models, and rank-preserving transforms are all viable |
| Weak signal expected | Regularisation is important; complex models will likely overfit |

---

## 4. Baseline Hypotheses

Stated before any EDA — to be tested in Stage 2:

1. **Momentum**: Recent positive return predicts near-term positive return
   (short-term momentum is documented in commodity markets).
2. **Mean-reversion**: Large recent moves predict reversal over longer horizons.
3. **Cross-commodity spillover**: Lagged returns from correlated commodities
   improve prediction of a given asset.
4. **Macro sensitivity**: Commodities cluster by macro exposure (risk-on/risk-off),
   and macro regime proxies carry predictive value.

---

## 5. Research Questions for Stage 2 EDA

- What is the time span and frequency of the dataset? How many commodities?
- What does the return distribution look like per commodity — fat tails, structural breaks?
- What is the autocorrelation structure of each commodity's returns?
- How correlated are commodities with each other across time?
- What is the naive Pearson baseline (zero prediction, last-period return)?
- Does missingness vary by commodity or period?

---

## 6. Stage 1 Deliverables

| File | Purpose |
|------|---------|
| `src/utils/paths.py` | Central path config; all data paths defined in one place |
| `src/data/load_data.py` | Typed data loading with missing-data warnings |
| `src/evaluation/metric.py` | Local metric before any leaderboard guessing |
| `tests/test_metric.py` | 24 unit tests covering correctness and edge cases |
| `docs/metric_explainer.md` | Plain-English and mathematical metric documentation |
| `docs/data_dictionary.md` | Field-by-field reference (inferred; to be updated with real data) |
| `docs/validation_strategy.md` | Walk-forward validation rationale and configuration |

---

## 7. What Stage 1 Does Not Do

- No model training.
- No feature engineering.
- No hyperparameter search.
- No submission file.
- No claims about leaderboard performance.
