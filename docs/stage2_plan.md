# Stage 2: EDA, Feature Engineering, and First Baseline

## Objective

Move from a working scaffold (Stage 1) to the first honest, reproducible baseline
score on held-out data. Stage 2 is complete when:

1. The 424 targets and 4 lag horizons are understood through EDA.
2. A walk-forward validation split is implemented and documented.
3. A simple baseline model produces a real Spearman-Sharpe score on the validation set.
4. The error analysis identifies the most actionable improvements for Stage 3.

---

## Notebook Sequence

| Notebook | Purpose | Status |
|----------|---------|--------|
| `01_data_and_metric_audit.ipynb` | Verify data loading, shapes, and metric implementations | Ready to run |
| `02_target_pairs_and_horizon_eda.ipynb` | EDA on the 424 targets across 4 lags | Ready to run |
| `03_validation_design.ipynb` | Walk-forward fold construction and split statistics | Ready to run |
| `04_baseline_features_and_model.ipynb` | First baseline: lag features + Ridge regression | Ready to run |
| `05_error_analysis_and_stability.ipynb` | Per-period and per-lag model diagnostics | Ready to run |
| `06_submission_pipeline_check.ipynb` | End-to-end submission pipeline check | Ready to run |

Run notebooks in order. Each calls `src/` functions; no business logic is defined inside notebooks.

---

## Baseline Modeling Scope

The first baseline is intentionally minimal. The goal is a **credible, reproducible
score** — not performance maximisation.

### Data split
- Train: `date_id` 0–1568 (approximately 80% of the 1961 training days)
- Validation: `date_id` 1569–1960 (remaining ~20%)
- No purge gap for the first baseline (features are simple lags, short lookback)

### Features (v0)
- Lagged values of key feature columns: 1, 2, 3, 5, and 10-day lags
- Cross-sectional z-score normalisation within each time step
- No interaction terms or rolling statistics in v0

### Model
- **Ridge regression**: one model per target (424 models total)
- Regularisation parameter: `alpha=1.0` (tune in Stage 3)
- Fit on training set only; predict on validation set

### Evaluation
- Primary: `spearman_sharpe_score(y_pred, y_true)` on validation period
- Secondary: `mean_row_pearson(y_pred, y_true)` for reference
- Breakdown by lag (1, 2, 3, 4) to identify which horizons are most predictable

### Baselines to beat
Before fitting any model, compute:
1. Zero-prediction baseline: predict 0 for all targets (Spearman = NaN/0)
2. Last-period return: predict the target value from the previous time step

Any trained model should beat both.

---

## Feature Engineering Roadmap (Stage 2 → Stage 3)

| Version | Features | Notes |
|---------|---------|-------|
| v0 | Raw lagged feature values (lag 1–10) | First baseline |
| v1 | Rolling means and standard deviations (5, 20-day) | Momentum and volatility |
| v2 | Cross-asset spread features (differences between related assets) | Pair-specific signals |
| v3 | Cross-sectional rank features within asset groups | Scale-invariant inputs |

---

## Decisions Deferred to Stage 3

- Hyperparameter tuning (Ridge alpha, tree depth)
- Multi-fold walk-forward validation (5+ folds)
- Non-linear models (LightGBM, XGBoost)
- Per-lag vs joint modelling
- Feature selection and importance analysis

---

## Stage 2 Completion Criteria

- [ ] All 6 notebooks run end-to-end without errors
- [ ] EDA notebook documents target return distributions and missingness patterns
- [ ] Validation split is defined and justified in notebook 03
- [ ] Baseline model produces a finite Spearman-Sharpe score on the validation set
- [ ] Error analysis identifies at least 3 actionable hypotheses for Stage 3
- [ ] README and this doc updated with actual baseline scores
