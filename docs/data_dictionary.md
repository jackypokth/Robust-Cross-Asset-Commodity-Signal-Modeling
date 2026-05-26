# Data Dictionary

> All fields marked **[CONFIRMED]** have been verified against the actual
> downloaded competition files.

---

## Files

| File | Rows | Cols | Description |
|------|------|------|-------------|
| `train.csv` | 1,961 | 558 | Training feature matrix |
| `train_labels.csv` | 1,961 | 425 | Training target values (424 targets) |
| `test.csv` | 134 | 559 | Test feature matrix (evaluation period) |
| `target_pairs.csv` | 424 | 3 | Maps each target to a lag and asset pair |
| `test_labels_lag_1.csv` | 134 | 108 | Realized test targets at lag 1 |
| `test_labels_lag_2.csv` | 134 | 108 | Realized test targets at lag 2 |
| `test_labels_lag_3.csv` | 134 | 108 | Realized test targets at lag 3 |
| `test_labels_lag_4.csv` | 134 | 108 | Realized test targets at lag 4 |

---

## `train.csv` [CONFIRMED]

Wide format: rows = trading days, columns = asset features.

| Column | Type | Description |
|--------|------|-------------|
| `date_id` | int | Integer day index, 0-based (range: 0–1960). Not a calendar date. |
| `LME_AH_Close` | float | LME Aluminium Alloy (LME_AH) closing price |
| `LME_CA_Close` | float | LME Copper (LME_CA) closing price |
| `LME_PB_Close` | float | LME Lead (LME_PB) closing price |
| `LME_ZS_Close` | float | LME Zinc (LME_ZS) closing price |
| `JPX_Gold_Mini_Futures_Open` | float | JPX Gold Mini futures open price |
| `JPX_Gold_Rolling-Spot_Futures_Open` | float | JPX Gold rolling-spot futures open price |
| `JPX_Platinum_*` | float | JPX Platinum futures (multiple contract types and OHLC fields) |
| `FX_*` | float | Currency pair rates (e.g. `FX_AUDJPY`, `FX_NOKGBP`, `FX_ZARCHF`) |
| `US_Stock_*_adj_close` | float | US ETF adjusted close prices (e.g. `US_Stock_VT_adj_close`, `US_Stock_VYM_adj_close`, `US_Stock_IEMG_adj_close`) |

**Shape:** 1,961 rows × 558 columns (1 index + 557 features)

**Missing values:** ~45,054 total. Expected — different exchanges have different trading calendars (e.g. LME vs JPX vs US equity market holidays).

### Feature categories

| Category | Examples | Count (approx) |
|----------|---------|----------------|
| LME metals | AH (Aluminium Alloy), CA (Copper), PB (Lead), ZS (Zinc) — OHLC | ~20 |
| JPX precious metals | Gold Mini, Gold Rolling-Spot, Gold Standard, Platinum Mini, Platinum Standard — OHLC | ~40 |
| FX rates | AUDJPY, NOKGBP, NOKCHF, NOKJPY, ZARCHF, ZARGBP, and many more | ~400+ |
| US equities | VT, VYM, IEMG adjusted close | ~10 |

---

## `train_labels.csv` [CONFIRMED]

Wide format: rows = trading days, columns = target values.

| Column | Type | Description |
|--------|------|-------------|
| `date_id` | int | Integer day index matching `train.csv` |
| `target_0` … `target_423` | float | Return of an asset or asset-pair at a given lag |

**Shape:** 1,961 rows × 425 columns (1 index + 424 targets)

The 424 targets are divided equally across 4 forecast lags:
- Lag 1: targets 0–105 (106 targets)
- Lag 2: targets 106–211 (106 targets)
- Lag 3: targets 212–317 (106 targets)
- Lag 4: targets 318–423 (106 targets)

---

## `test.csv` [CONFIRMED]

Same structure as `train.csv`.

| Column | Type | Description |
|--------|------|-------------|
| `date_id` | int | Integer day index (range: 1827–1960) |
| feature columns | float | Same feature set as training, plus one additional column |

**Shape:** 134 rows × 559 columns

**Note:** The test `date_id` range (1827–1960) falls within the training date range (0–1960). The test labels are not included in `train_labels.csv` — they are provided separately in the `test_labels_lag_*.csv` files for local evaluation.

---

## `target_pairs.csv` [CONFIRMED]

Maps each of the 424 target column names to a forecast lag and an asset-pair description.

| Column | Type | Description |
|--------|------|-------------|
| `target` | str | Target column name (e.g. `target_0`) |
| `lag` | int | Forecast horizon in trading days (1, 2, 3, or 4) |
| `pair` | str | Asset or asset-pair description (e.g. `LME_AH_Close - LME_ZS_Close`) |

**Shape:** 424 rows × 3 columns

### Target types

| Type | Example pair | Notes |
|------|-------------|-------|
| Single asset | `US_Stock_VT_adj_close` | Return of one asset at the given lag |
| Asset pair (difference) | `LME_AH_Close - LME_ZS_Close` | Return difference between two assets |

The `-` separator in the pair string indicates a pairwise return difference. This is a **relative value / spread** prediction task, not a single-asset return prediction.

### Example rows

| target | lag | pair |
|--------|-----|------|
| target_0 | 1 | US_Stock_VT_adj_close |
| target_1 | 1 | LME_PB_Close - US_Stock_VT_adj_close |
| target_2 | 1 | LME_CA_Close - LME_ZS_Close |
| target_3 | 1 | LME_AH_Close - LME_ZS_Close |
| target_4 | 1 | LME_AH_Close - JPX_Gold_Standard_Futures_Close |

---

## `test_labels_lag_N.csv` [CONFIRMED]

Realized target values for the test evaluation period, at a specific lag horizon.

| Column | Type | Description |
|--------|------|-------------|
| `date_id` | int | Day index, shifted forward by the lag (range: 1829–1962 for lag 1) |
| `target_0` … | float | Realized returns for the corresponding lag targets |

**Shape:** 134 rows × 108 columns (1 index + 107 targets per file)

**Note:** Each lag file contains only the 106–107 targets for that specific lag. The `date_id` values in these files are shifted relative to the test feature `date_id` values by the forecast horizon.

---

## Target Variable — Key Modelling Notes

**What is being predicted:**
The `target` at lag L for pair "Asset_A - Asset_B" is:

```
target = return(Asset_A, t+L) - return(Asset_B, t+L)
```

where return is likely a log-return or normalized % change computed over the next L trading days.

**Why this matters:**
- This is a cross-sectional, **relative** prediction problem — not absolute return forecasting.
- The Pearson correlation metric is computed row-wise (per date_id), then averaged. This rewards ranking assets correctly within each period.
- Absolute scale of predictions does not affect the score; sign and ranking do.
- Models that predict spreads tend to benefit from features that capture cross-asset divergence (basis, momentum differentials, FX exposure differences).

---

## Update Log

| Stage | Entry |
|-------|-------|
| Stage 1 | Initial draft — all fields inferred, awaiting data download |
| Stage 2 | Full rewrite — all fields confirmed against downloaded competition files |
