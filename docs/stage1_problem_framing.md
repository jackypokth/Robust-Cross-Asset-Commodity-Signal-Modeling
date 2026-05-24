# Stage 1: Problem Framing

## Objective

Understand the competition from first principles before writing any modeling code. This document answers:

1. What exactly is being predicted?
2. How is success measured?
3. What structure does the data have?
4. What modeling approach does this suggest?
5. What are the early research questions?

---

## 1. What Is Being Predicted?

The competition asks participants to predict **future returns of commodity assets** across multiple time steps. A "return" here is the percentage change in price from one period to the next (or from now to a future horizon).

Key framing points:
- We are not predicting raw prices — we predict **returns** (or signals correlated with returns).
- The evaluation metric (Pearson correlation) means the **ranking** of predicted returns matters, not their absolute scale.
- Predictions likely span multiple commodities simultaneously, creating a **cross-sectional dimension**.

---

## 2. Data Structure (Inferred)

Based on the competition framing, the dataset likely has this structure:

```
date | commodity_id | feature_1 | feature_2 | ... | target
```

- **date**: Observation timestamp (daily, weekly, or monthly frequency).
- **commodity_id**: Identifier for the commodity (e.g., crude oil, copper, corn).
- **features**: Lagged price information, technical indicators, macro variables, or encoded signals.
- **target**: Future return (e.g., next-period log return or raw % change).

> **Assumption**: The dataset is a panel (long-format time series with multiple assets). This needs verification after downloading competition data.

---

## 3. Modeling Implications

| Observation | Implication |
|-------------|-------------|
| Return prediction with Pearson metric | Scale doesn't matter; rank signal does |
| Panel data (multiple commodities × time) | Cross-sectional models can exploit relative patterns |
| Financial time series | Strong temporal autocorrelation, regime shifts |
| Ranking-aware metric | Tree models, linear regression, and rank-preserving transforms all viable |
| Weak signal expected | Regularization critical; complex models may overfit |

---

## 4. Baseline Hypotheses

Before any EDA, we state falsifiable hypotheses:

1. **Momentum hypothesis**: Recent positive return predicts near-term positive return (short-term momentum effect is documented in commodities).
2. **Mean-reversion hypothesis**: Large recent moves predict reversal over longer horizons.
3. **Cross-commodity correlation**: A basket of lagged returns from correlated commodities improves prediction of any single commodity.
4. **Macro sensitivity**: Commodities cluster by macro sensitivity (risk-on/risk-off), and macro regime proxies have predictive value.

---

## 5. Initial Research Questions

These should be answered during EDA (Stage 2):

- What is the time span of the dataset? How many commodities?
- What is the return distribution per commodity? Are there fat tails, structural breaks?
- What is the autocorrelation structure of each commodity's returns?
- How correlated are commodities with each other over time?
- Does the target exhibit any seasonal patterns?
- What is the baseline Pearson correlation of a naive model (e.g., predict zero, or predict last return)?
- Does the number of usable training examples change significantly over time (missing data patterns)?

---

## 6. What Stage 1 Delivers

| Deliverable | Purpose |
|-------------|---------|
| `src/utils/paths.py` | Central path config so all scripts find data consistently |
| `src/data/load_data.py` | Clean, typed data loading functions |
| `src/evaluation/metric.py` | Local metric implementation before leaderboard guesswork |
| `tests/test_metric.py` | Validates metric correctness on synthetic examples |
| `docs/metric_explainer.md` | Plain-English metric documentation |
| `docs/data_dictionary.md` | Field-by-field data reference |
| `docs/validation_strategy.md` | Time-series validation plan |

---

## 7. What Stage 1 Explicitly Does NOT Do

- No model training
- No feature engineering
- No hyperparameter search
- No submission file generation
- No claims about leaderboard performance
