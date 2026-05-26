# MITSUI&CO. Commodity Prediction Challenge

Kaggle competition: predict cross-sectional relative returns of commodity assets.
Evaluated by mean row-wise Pearson correlation across time steps.

---

## Problem

Given historical price and feature data for a set of commodities, predict which
commodities will outperform or underperform each other at each future time step.
This is a ranking problem: the metric is Pearson correlation between predicted
and actual returns, not absolute return magnitude.

See [`docs/stage1_problem_framing.md`](docs/stage1_problem_framing.md) for the
full problem breakdown, baseline hypotheses, and open research questions.

---

## Repository Structure

```
.
├── data/raw/               # Competition CSVs — gitignored, download from Kaggle
├── docs/                   # Research notes and stage documentation
│   ├── project_overview.md
│   ├── stage1_problem_framing.md
│   ├── metric_explainer.md
│   ├── data_dictionary.md
│   └── validation_strategy.md
├── notebooks/              # Exploratory analysis (calls src/, never defines logic)
│   └── 01_metric_and_data_understanding.md
├── src/
│   ├── data/load_data.py       # Data loading and wide-format pivot
│   ├── evaluation/metric.py    # Local metric implementation (Pearson, row-wise)
│   └── utils/paths.py          # Centralised path config
├── tests/
│   └── test_metric.py      # 24 unit tests for the metric
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

## Design Decisions

- **Metric-first**: `metric.py` is implemented and fully tested before any model code exists.
- **No random K-fold**: all validation uses walk-forward (temporal) splits.
  See [`docs/validation_strategy.md`](docs/validation_strategy.md).
- **`src/` over notebooks**: exploratory notebooks call `src/` functions — they never define logic.
- **Honest assumptions**: inferred data fields are flagged as `[INFERRED]` until verified
  against downloaded data.

---

## Progress

| Stage | Status | Description |
|-------|--------|-------------|
| 1 | Complete | Problem framing, metric implementation, data loading scaffold, validation strategy |
| 2 | Planned | EDA, feature engineering, baseline models |
| 3 | Planned | Walk-forward model selection |
| 4 | Planned | Ensembling, final submission |

---

## Notes

- Competition data is not committed — place downloaded files in `data/raw/`.
- Data field names are inferred until the competition CSVs are inspected.
  See [`docs/data_dictionary.md`](docs/data_dictionary.md).

---

## Suggested GitHub About

> Kaggle ML pipeline for commodity return prediction. Ranking-aware metric, walk-forward validation, no random K-fold.

**Topics:** `kaggle`, `machine-learning`, `time-series`, `financial-ml`, `commodities`, `python`
