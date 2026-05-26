# Data Dictionary

> Fields marked **[INFERRED]** are based on the competition description and common
> conventions for Kaggle financial prediction tasks. They must be verified after
> downloading the competition data. Fields marked **[CONFIRMED]** have been verified
> against the actual files.

---

## Files

| File | Description |
|------|-------------|
| `train.csv` | Historical observations with target values for training |
| `test.csv` | Observations for which predictions must be submitted |
| `sample_submission.csv` | Required submission format |

---

## `train.csv`

| Column | Type | Description | Status |
|--------|------|-------------|--------|
| `date` | date | Observation date (YYYY-MM-DD) | [INFERRED] Frequency TBD |
| `commodity` | string | Asset identifier | [INFERRED] |
| `target` | float | Future return to predict | [INFERRED] Log-return or % change |
| `feature_*` | float | Input features | [INFERRED] Count and names TBD |

---

## `test.csv`

Same structure as `train.csv`, without `target`.

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Observation date |
| `commodity` | string | Asset identifier |
| `feature_*` | float | Input features |

---

## `sample_submission.csv`

| Column | Type | Description |
|--------|------|-------------|
| `id` | string/int | Row identifier matching test rows |
| `target` | float | Predicted return |

---

## Target Variable

The target is a **future commodity return**. Key points for modeling:

- Return horizon: likely 1-period ahead (verify from competition page).
- The metric (Pearson correlation) is scale-invariant — the exact return units do
  not affect the score.
- Log returns have better Gaussian properties; % returns have heavier tails.

---

## Commodity Universe (Expected)

Competition likely covers a mix of:

| Category | Examples |
|----------|---------|
| Energy | Crude oil (WTI, Brent), natural gas |
| Metals | Gold, silver, copper, aluminum |
| Agriculture | Corn, wheat, soybeans |

> Verify the exact list from `train.csv` after downloading.

---

## Feature Categories (Expected)

| Category | Examples |
|----------|---------|
| Lagged returns | 1-period, 5-period, 20-period lagged returns |
| Volatility | Rolling standard deviation of returns |
| Technical | Moving averages, RSI (if included) |
| Cross-asset | Returns of correlated commodities |
| Macro | Dollar index, interest rate proxies (if included) |

---

## Update Log

| Stage | Entry |
|-------|-------|
| Stage 1 | Initial draft — all fields inferred, awaiting data download |
| Stage 2 | *(to be filled after EDA)* |
