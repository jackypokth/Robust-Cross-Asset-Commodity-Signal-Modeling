# MITSUI&CO. Commodity Prediction Challenge

A machine learning research repository for the [MITSUI&CO. Commodity Prediction Challenge](https://www.kaggle.com/competitions/mitsui-co-commodity-prediction-challenge) on Kaggle.

This repository is designed as a **research-grade, reproducible ML project** — not a notebook dump. Logic lives in `src/`, experiments are tracked cleanly, and documentation is written for both collaborators and interviewers.

---

## Competition Summary

Predict the **direction and magnitude of future returns** for a set of commodity assets across multiple forecast horizons. The evaluation metric rewards **ranking quality** via Pearson correlation between predicted and realized returns, averaged across time periods.

See [`docs/stage1_problem_framing.md`](docs/stage1_problem_framing.md) for the full problem breakdown.

---

## Repository Structure

```
.
├── data/                   # Raw and processed data (gitignored)
│   ├── raw/
│   └── processed/
├── docs/                   # Research documentation
│   ├── project_overview.md
│   ├── stage1_problem_framing.md
│   ├── metric_explainer.md
│   ├── data_dictionary.md
│   └── validation_strategy.md
├── notebooks/              # Exploratory analysis (not reusable logic)
│   └── 01_metric_and_data_understanding.md
├── src/                    # Reusable source code
│   ├── data/               # Data loading and preprocessing
│   ├── evaluation/         # Metric implementations
│   └── utils/              # Path management, helpers
└── tests/                  # Unit tests for src/ modules
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Load data and inspect
python -c "from src.data.load_data import load_train; df = load_train(); print(df.shape)"
```

---

## Research Stages

| Stage | Status | Description |
|-------|--------|-------------|
| 1 | In progress | Problem framing, data understanding, metric implementation |
| 2 | Planned | Feature engineering, baseline models |
| 3 | Planned | Time-series cross-validation, model selection |
| 4 | Planned | Ensembling and final submission |

---

## Key Design Decisions

- **No random K-fold**: All validation respects temporal ordering.
- **Metric-first**: We implement the evaluation metric locally before training any model.
- **src/ over notebooks**: Exploratory notebooks call `src/` functions — they do not define them.
- **No premature abstraction**: Config-driven pipelines come in Stage 3+, not Stage 1.

---

## Notes and Assumptions

See individual docs for flagged assumptions. Competition data is not committed to this repo — place downloaded files in `data/raw/`.
