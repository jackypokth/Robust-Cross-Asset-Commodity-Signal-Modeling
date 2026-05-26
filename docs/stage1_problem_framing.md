# Stage 1: Problem Framing

## Objective

Establish a clear understanding of the competition before writing any modeling
code. This document records:

1. What is being predicted and how success is measured.
2. What structure the data actually has (confirmed against downloaded files).
3. What the modeling implications are.
4. Falsifiable baseline hypotheses to test in Stage 2.
5. Open research questions to answer during EDA.

---

## 1. What Is Being Predicted?

The competition asks for **future returns of 424 target spreads** at each time step.
Targets are either:
- A single asset return: e.g. `US_Stock_VT_adj_close`
- A pairwise return difference: e.g. `LME_AH_Close - LME_ZS_Close`

Each target is defined at a specific forecast horizon: lag 1, 2, 3, or 4 trading days.
There are 106 targets per lag, giving 424 targets total.

Key framing:
- This is a **cross-sectional spread ranking** problem, not a univariate return
  forecasting problem.
- The evaluation metric uses **Spearman rank correlation** — the ordering of predicted
  spread returns matters, not their magnitude.
- Predictions cover all 424 targets simultaneously at each time step, giving a rich
  cross-sectional structure to exploit.

---

## 2. Confirmed Data Structure

All column names and shapes have been verified against the downloaded competition files.

### Features — `train.csv`

Wide format: rows = trading days, columns = asset features.

- **Shape:** 1,961 rows × 558 columns
- **Index column:** `date_id` — integer day counter (0–1960), not a calendar date
- **Feature categories:**
  - LME metals: Aluminium Alloy (AH), Copper (CA), Lead (PB), Zinc (ZS) — OHLC prices
  - JPX precious metals: Gold (Mini, Rolling-Spot, Standard) and Platinum — OHLC futures
  - FX rates: ~400 currency pair rates (AUDJPY, NOKGBP, ZARCHF, and many more)
  - US equity ETFs: VT, VYM, IEMG adjusted close prices
- **Missing values:** ~45,054 total — expected due to exchange calendar mismatches

### Labels — `train_labels.csv`

Wide format: rows = trading days, columns = target values.

- **Shape:** 1,961 rows × 425 columns
- **Index column:** `date_id` — matches `train.csv`
- **Target columns:** `target_0` through `target_423` (424 targets)
- **Lag distribution:** 106 targets per lag × 4 lags = 424 total

### Target Metadata — `target_pairs.csv`

- **Shape:** 424 rows × 3 columns
- **Columns:** `target` (name), `lag` (1–4), `pair` (asset or asset-pair description)
- Single-asset targets have no separator; pair targets use ` - ` separator

### Test Features — `test.csv`

- **Shape:** 134 rows × 559 columns
- **date_id range:** 1827–1960 (the final 134 trading days of the training period)
- Note: one additional feature column vs training data

### Test Labels — `test_labels_lag_N.csv`

- Separate file per lag (1, 2, 3, 4)
- **Shape:** 134 rows × 108 columns each
- `date_id` values are offset from test feature `date_id` by the lag horizon

---

## 3. Modeling Implications

| Observation | Implication |
|-------------|-------------|
| Spearman-Sharpe metric | Scale is irrelevant; correct ranking AND consistency matter |
| 424 targets simultaneously | Cross-sectional models can exploit relative patterns across targets |
| Wide-format, integer date_id | No calendar features directly available; use date_id offsets for lags |
| 4 forecast horizons | Models for lag 1 and lag 4 may behave very differently; consider per-lag models |
| Pairwise spread targets | Features capturing relative momentum between pairs may outperform single-asset signals |
| ~45k missing values | LME and JPX calendars differ; missingness is structured, not random |
| ~557 features, 1961 rows | Regularisation is essential; high-dimensional regime |

---

## 4. Baseline Hypotheses

Stated before any EDA — to be tested in Stage 2:

1. **Cross-asset momentum**: Recent positive return in one asset predicts
   positive return in a spread where that asset is the long leg.
2. **Mean-reversion in spreads**: Large recent spread moves predict reversal
   over longer horizons (especially at lag 3 and 4).
3. **FX-commodity linkage**: FX rates for commodity-exporting countries
   (AUD, NOK, ZAR) carry predictive information for LME metal spreads.
4. **Lag correlation**: Targets at short lags (1, 2) are more predictable than
   long lags (3, 4) due to autocorrelation decay.

---

## 5. Research Questions for Stage 2 EDA

- What is the autocorrelation structure of spread returns at each lag?
- Which feature groups (LME, JPX, FX, US equity) have the highest predictive power?
- How correlated are targets within the same lag vs across lags?
- What is the naive Spearman-Sharpe baseline (zero predictions, last-period spread)?
- Does missingness cluster by exchange or time period?
- Are there structural breaks in the feature or target distributions over date_id?

---

## 6. Stage 1 Deliverables

| File | Purpose | Status |
|------|---------|--------|
| `src/utils/paths.py` | Central path config; all data paths defined here | Complete |
| `src/data/load_data.py` | Wide-format data loading with missing-data warnings | Complete |
| `src/evaluation/metric.py` | Spearman-Sharpe (official) + Pearson proxy, fully tested | Complete |
| `tests/test_metric.py` | Unit tests covering both metric implementations | Complete |
| `docs/metric_explainer.md` | Explanation of official metric and proxy | Complete |
| `docs/data_dictionary.md` | Confirmed field-by-field reference | Complete |
| `docs/validation_strategy.md` | Walk-forward validation rationale | Complete |

---

## 7. What Stage 1 Does Not Do

- No model training.
- No feature engineering.
- No hyperparameter search.
- No submission file.
- No claims about leaderboard performance.
