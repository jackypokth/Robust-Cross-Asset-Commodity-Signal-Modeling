# Project Overview

## Competition

**Name:** MITSUI&CO. Commodity Prediction Challenge
**Host:** Mitsui & Co., Ltd. (via Kaggle)
**Task type:** Supervised time-series regression / ranking
**Domain:** Financial ML — commodity markets

---

## Business Context

Mitsui & Co. is one of Japan's largest trading conglomerates with major exposure to commodity markets (energy, metals, agriculture). The competition reflects a real-world problem: **predicting near-future commodity price movements** to inform trading and risk management decisions.

This is not a pure price-level prediction task. The goal is to predict **cross-sectional relative performance** of commodities — i.e., which commodities will go up or down relative to each other — rather than predicting absolute price levels. This framing makes it a **ranking problem** as much as a regression problem.

---

## Prediction Task

- **Input:** Historical price data, potentially supplemented by macroeconomic indicators, market signals, or other commodity-related features.
- **Output:** Predicted future returns (or return ranks) for each commodity at each time step.
- **Horizon:** Multi-step ahead forecasting across one or more prediction windows.

---

## Why This Problem Is Hard

1. **Non-stationarity**: Commodity prices are driven by shifting supply/demand dynamics, geopolitical events, and weather — distributions change over time.
2. **Low signal-to-noise ratio**: Financial return series are notoriously noisy; genuine predictive signal is weak.
3. **Cross-sectional structure**: Predictions must be good in a *relative* sense, not just absolute — ranking aware metrics penalize systematic bias less than directional errors.
4. **Temporal leakage risk**: Standard cross-validation splits will leak future information and produce overly optimistic estimates.
5. **Limited sample size**: Financial time-series datasets are rarely large by ML standards.

---

## Repository Philosophy

| Principle | Implementation |
|-----------|---------------|
| Reproducibility | Fixed seeds, versioned data paths, explicit configs |
| Separation of concerns | I/O in `src/data/`, logic in `src/evaluation/`, utils separate |
| Test-driven metric | Metric implemented and tested before training |
| Honest evaluation | No random K-fold; walk-forward only |
| Documentation-first | Docs written alongside code, not after |

---

## Team and Timeline

- Solo / research project
- Stage 1 (this stage): problem framing and tooling setup
- Target: submission-ready pipeline by competition deadline
