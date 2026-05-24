# Data Dictionary

> **Status**: This dictionary is partially inferred from competition description and standard Kaggle commodity challenge conventions. Fields marked [INFERRED] must be verified after downloading competition data. Fields marked [CONFIRMED] are from the official competition page.

---

## Files

| File | Description |
|------|-------------|
| `train.csv` | Historical observations with targets for model training |
| `test.csv` | Observations for which predictions must be submitted |
| `sample_submission.csv` | Correct submission format |
| `supplemental_data.csv` | Additional contextual data (if provided) |

---

## `train.csv` — Field Reference

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `date` | date | Observation date (YYYY-MM-DD format) | [INFERRED] Frequency TBD — daily, weekly, or monthly |
| `commodity` | string | Commodity identifier (e.g., "crude_oil", "copper") | [INFERRED] Categorical |
| `target` | float | Future return to predict | [INFERRED] Likely log-return or % change |
| `feature_*` | float | Engineered or raw input features | [INFERRED] Count and meaning TBD |

---

## `test.csv` — Field Reference

Same structure as `train.csv` but without the `target` column.

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Observation date |
| `commodity` | string | Commodity identifier |
| `feature_*` | float | Input features |

---

## `sample_submission.csv` — Field Reference

| Column | Type | Description |
|--------|------|-------------|
| `id` | string or int | Row identifier matching test rows |
| `target` | float | Your predicted return for this row |

---

## Target Variable

The target is the **future return** of a commodity. Specific assumptions (to verify):

- Return horizon: likely 1-period ahead (but may be multi-horizon)
- Return type: percentage change or log return
- Return direction: positive = commodity appreciated; negative = depreciated

**Why this matters for modeling:**
- Log returns are better for Gaussian-assumption models (e.g., linear regression)
- % returns are common in practice and may have heavier tails
- The metric is Pearson correlation, so the precise scale doesn't affect scoring

---

## Commodities (Expected)

Standard commodity prediction competitions typically include assets from these categories:

| Category | Examples |
|----------|---------|
| Energy | Crude Oil (WTI, Brent), Natural Gas, Heating Oil |
| Metals | Gold, Silver, Copper, Aluminum, Platinum |
| Agriculture | Corn, Wheat, Soybeans, Sugar, Cotton |
| Soft commodities | Coffee, Cocoa |

> Verify the exact list from `train.csv` after downloading data.

---

## Feature Categories (Inferred)

Based on common financial ML competitions, features likely include:

| Category | Description |
|----------|-------------|
| Price history | Lagged returns at various horizons (1d, 5d, 20d, 60d) |
| Technical indicators | Moving averages, RSI, Bollinger Bands, MACD |
| Volatility | Rolling standard deviation of returns |
| Volume | Trading volume, open interest (if futures data) |
| Cross-commodity | Returns of correlated commodities |
| Macro | Interest rates, dollar index, risk sentiment proxies |

---

## Data Update Log

| Date | Update |
|------|--------|
| Stage 1 | Initial draft — fields inferred from competition context |
| TBD | Update after downloading and inspecting actual data |
