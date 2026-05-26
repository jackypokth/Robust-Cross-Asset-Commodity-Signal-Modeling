# MITSUI&CO. Commodity Prediction Challenge

Kaggle competition: predict cross-sectional relative returns of commodity asset pairs.
Evaluated by a **Spearman-Sharpe score** — mean row-wise Spearman rank correlation
divided by its standard deviation across time steps.

---

## Problem

Given historical price data for LME metals, JPX precious metals futures, FX rates,
and US equity ETFs, predict which of 424 target spreads (e.g. `LME_AH_Close -
LME_ZS_Close`) will outperform or underperform at each future time step.

The dataset is wide-format: 1,961 training days × 557 features, with 424 target
columns spanning 4 forecast lags (1, 2, 3, 4 days). Targets are pairwise return
differences between asset pairs — this is a cross-sectional spread ranking problem.

See [`docs/stage1_problem_framing.md`](docs/stage1_problem_framing.md) for the full
problem breakdown, baseline hypotheses, and research questions.

---

## Repository Structure

```
.
├── data/raw/               # Competition CSVs — gitignored, download from Kaggle
├── docs/
│   ├── stage1_problem_framing.md   # Problem framing and baseline hypotheses
│   ├── stage2_plan.md              # Stage 2 active roadmap
│   ├── metric_explainer.md         # Official metric (Spearman-Sharpe) and proxy
│   ├── data_dictionary.md          # Confirmed column/file reference
│   └── validation_strategy.md      # Walk-forward validation design
├── notebooks/
│   ├── 01_data_and_metric_audit.ipynb          # Data integrity + metric verification
│   ├── 02_target_pairs_and_horizon_eda.ipynb   # EDA on the 424 targets and 4 lags
│   ├── 03_validation_design.ipynb              # Walk-forward fold construction
│   ├── 04_baseline_features_and_model.ipynb    # First honest baseline
│   ├── 05_error_analysis_and_stability.ipynb   # Where and why the baseline fails
│   └── 06_submission_pipeline_check.ipynb      # End-to-end submission check
├── src/
│   ├── data/load_data.py       # Wide-format data loading (train, labels, test)
│   ├── evaluation/metric.py    # Spearman-Sharpe (official) + Pearson (proxy)
│   └── utils/paths.py          # Centralised path config
├── tests/
│   └── test_metric.py          # Unit tests for both metric implementations
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -v

# Verify data files are present after downloading from Kaggle
python src/utils/paths.py
```

---

## Pipeline

```
data/raw/
  train.csv           (1961 × 558)   features
  train_labels.csv    (1961 × 425)   targets (424 targets, 4 lags)
  target_pairs.csv    (424 × 3)      target → (lag, asset pair) mapping
  test.csv            (134 × 559)    evaluation features
  test_labels_lag_*.csv              held-out labels for local eval

src/data/load_data.py
  load_train()        → wide feature DataFrame
  load_labels()       → wide label DataFrame (424 targets)
  load_target_pairs() → lag + pair metadata
  get_labels_for_lag(labels, pairs, lag)  → single-lag slice

src/evaluation/metric.py
  spearman_sharpe_score(y_pred, y_true)   ← official competition metric
  mean_row_pearson(y_pred, y_true)        ← offline proxy (fast iteration)

notebooks/
  01 → verify data and metric
  02 → EDA on targets and lags
  03 → walk-forward fold design
  04 → first baseline (features + model)
  05 → error analysis
  06 → submission check
```

---

## Design Decisions

- **Metric-first**: both the official Spearman-Sharpe metric and a Pearson proxy are
  implemented and fully tested before any model code exists.
- **No random K-fold**: all validation uses walk-forward (temporal) splits.
  See [`docs/validation_strategy.md`](docs/validation_strategy.md).
- **`src/` over notebooks**: exploratory notebooks call `src/` functions — they never
  define reusable logic.
- **Honest baselines**: no fabricated results; all scores come from real held-out data
  using the official metric.

---

## Progress

| Stage | Status | Description |
|-------|--------|-------------|
| 1 | Complete | Problem framing, metric implementation, data loading, validation strategy |
| 2 | Active | EDA, feature engineering, first baseline model |
| 3 | Planned | Walk-forward model selection across multiple folds |
| 4 | Planned | Ensembling, final submission |
