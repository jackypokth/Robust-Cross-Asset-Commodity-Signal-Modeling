# MITSUI&CO. Commodity Prediction Challenge

Machine learning research repository for the Kaggle commodity prediction competition. Predicts cross-sectional relative returns of commodity assets. Evaluated via mean row-wise Pearson correlation.

## Run & Operate

- `python3 -m pytest tests/ -v` — run all unit tests
- `python3 src/utils/paths.py` — verify data paths and check which files are present
- `pnpm --filter @workspace/api-server run dev` — run the API server (not needed for this ML project)
- `pnpm run typecheck` — full typecheck across all packages

## Stack

- Python 3.11
- Data: numpy, pandas, scipy
- Testing: pytest
- pnpm workspaces (monorepo shell), Node.js 24, TypeScript 5.9

## Where things live

- `README.md` — project overview for GitHub / interviewers
- `docs/` — research documentation (metric, data dict, validation strategy, stage framing)
- `src/evaluation/metric.py` — local Pearson correlation metric implementation
- `src/data/load_data.py` — typed data loading functions
- `src/utils/paths.py` — central path config (all data paths defined here)
- `tests/test_metric.py` — 24 unit tests covering the metric implementation
- `notebooks/01_metric_and_data_understanding.md` — notebook plan (convert to .ipynb after downloading data)
- `data/raw/` — place downloaded Kaggle data here (gitignored)
- `requirements.txt` — Python dependencies

## Architecture decisions

- **Metric-first development**: metric.py is implemented and fully tested before any model code exists.
- **No random K-fold**: all validation will use walk-forward (temporal) splits — documented in `docs/validation_strategy.md`.
- **src/ over notebooks**: exploratory notebooks call `src/` functions; they never define reusable logic.
- **Scale-invariance**: metric implementation correctly handles constant predictions (NaN) and respects Pearson's scale/mean-shift invariance.
- **No premature abstraction**: config-driven experiment pipelines are deferred to Stage 3+.

## Product

ML research pipeline for commodity return prediction. Stage 1 complete: problem framing, metric implementation, data loading scaffold, validation strategy, and documentation.

## User preferences

- ML-first: prioritize clean Python modules over frameworks or web app abstractions.
- Readable code with docstrings; functions small and testable.
- No fake results or placeholder claims — flag assumptions explicitly.
- Use pathlib for all file paths.

## Gotchas

- Data files go in `data/raw/` — they are gitignored. Check presence with `python3 src/utils/paths.py`.
- The metric averages **per-period** Pearson correlations — do not optimize MSE as a proxy.
- Walk-forward validation only — see `docs/validation_strategy.md` for gap/embargo sizing.

## Pointers

- See `docs/stage1_problem_framing.md` for current research questions and baseline hypotheses.
- See `docs/metric_explainer.md` for metric math and common pitfalls.
- See `docs/validation_strategy.md` for walk-forward vs. rolling window trade-offs.
