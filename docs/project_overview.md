# Project Overview

## Competition

**Name:** MITSUI&CO. Commodity Prediction Challenge
**Host:** Mitsui & Co., Ltd. (via Kaggle)
**Task type:** Supervised time-series regression / ranking
**Domain:** Financial ML — commodity markets

---

## Problem Statement

The task is to predict **future returns of a set of commodity assets** at each
time step, where success is measured by how well the predicted returns *rank*
the commodities relative to each other — not by the accuracy of the absolute
return magnitudes.

This framing has direct modeling implications: the evaluation metric (Pearson
correlation) is scale- and mean-shift-invariant within each time step. A model
that correctly identifies relative winners and losers scores the same regardless
of whether its predicted values are in percentage points or arbitrary units.

---

## Why This Problem Is Hard

1. **Non-stationarity**: Commodity markets are driven by macro shocks, geopolitical
   events, and supply dynamics — the generating distribution shifts over time.
2. **Low signal-to-noise**: Financial return series are notoriously noisy. Genuine
   predictive signal is weak and unstable across regimes.
3. **Cross-sectional structure**: A good prediction at time $t$ must correctly order
   *all* assets simultaneously, not just predict each one independently.
4. **Temporal leakage**: Standard K-fold cross-validation leaks future information
   into training, producing unrealistically optimistic validation scores.
5. **Small effective sample size**: Financial datasets have long time spans but
   limited independent observations once autocorrelation is accounted for.

---

## Repository Philosophy

| Principle | Implementation |
|-----------|---------------|
| Reproducibility | Fixed seeds, centralised path config, explicit data download instructions |
| Separation of concerns | I/O in `src/data/`, metric logic in `src/evaluation/`, paths in `src/utils/` |
| Metric-first | Metric implemented and unit-tested before any model is trained |
| Honest evaluation | No random K-fold; walk-forward splits only |
| No overclaiming | Inferred data fields are flagged; no fabricated results |

---

## Project Status

- **Stage 1 (complete):** Problem framing, metric implementation, data loading
  scaffold, validation strategy documentation.
- **Stage 2 (planned):** EDA, feature engineering, baseline models.
- **Stage 3+ (planned):** Walk-forward model selection, ensembling, submission.
