# Resume bullets and LinkedIn text

Everything below is true of this project. Keep the numbers; they show rigour. Do **not** claim the system predicts the
market or "beats the market" - it does not, and interviewers will probe.

## Resume bullets (pick 4-5)

**Stock Market Prediction & Analysis System** | Python, PyTorch, XGBoost, FastAPI, Streamlit, Docker | GitHub link

* Built an end-to-end, leakage-safe ML pipeline for next-day stock price and direction forecasting: 44 causal technical
  features, chronological 70/15/15 splits with a boundary gap, train-only scaling, and automated look-ahead tests.
* Implemented and compared baselines, XGBoost, an LSTM and a Transformer encoder (PyTorch) with early stopping, LR scheduling,
  checkpointing and Optuna tuning; evaluated with Wilson/bootstrap confidence intervals, binomial significance tests and
  Platt calibration - finding **no statistically significant edge** (best test AUC 0.546, 95% CI [0.493, 0.598]) and
  reporting the negative result transparently.
* Developed a cost-aware backtester (next-open execution, turnover costs, Sharpe, max drawdown, win rate) with a
  random-strategy null test and Bonferroni correction; ML signals did not beat buy-and-hold (+54.1%) after 5 bps costs and
  were indistinguishable from random strategies of equal exposure (p = 0.16).
* Shipped a FastAPI service (prediction, background training jobs, metrics; pydantic validation, typed error handling) and a
  Streamlit dashboard sharing one tested service layer; responses expose held-out evidence and never present raw scores as probabilities.
* Containerised with Docker/Compose (CPU-only PyTorch, non-root) and added GitHub Actions CI; ~130 pytest tests including
  leakage, backtest-timing, inference-parity and API/dashboard tests.

### Shorter version (3 bullets)

* Built a leakage-safe stock forecasting system comparing baselines, XGBoost, LSTM and Transformer models (PyTorch), with
  chronological splits, confidence intervals and significance tests; found no statistically significant edge and documented why.
* Wrote a cost-aware backtester (Sharpe, drawdown, random-strategy null test); ML strategies did not beat buy-and-hold (+54.1%) after costs.
* Deployed via FastAPI, Streamlit, Docker and GitHub Actions CI with ~130 automated tests.

### Optional one-liner for a skills/projects section

"Time-series ML with rigorous evaluation: leakage prevention, calibration, significance testing, backtesting, deployment."

## LinkedIn project description

**Stock Market Prediction & Analysis System - an end-to-end ML project where the honest answer was "no edge"**

I built a production-style machine-learning system that tries to forecast next-day stock price and direction with baselines,
XGBoost, an LSTM and a Transformer, and then asks the harder question: does it hold up? The pipeline uses 44 causal technical
features, strictly chronological splits, train-only scaling, early stopping and Optuna tuning, plus confidence intervals,
significance tests, calibration and a cost-aware backtest with a random-strategy null test.

The result: none of the models showed a statistically significant edge over simple baselines (best test AUC 0.546,
95% CI 0.493-0.598), and after trading costs no ML strategy beat buy-and-hold. Along the way I caught and fixed real issues -
a partial trading-day bar leaking into data, a calibration step that silently flipped a model's direction, and importance
scores that counted discarded trees - each now covered by tests.

The system includes a FastAPI backend, a Streamlit dashboard, Docker/Compose and GitHub Actions CI with about 130 automated tests.
It is an experimental, educational project - not financial advice.

Stack: Python, pandas, scikit-learn, XGBoost, PyTorch, Optuna, SHAP, FastAPI, Streamlit, Docker, GitHub Actions.

GitHub: https://github.com/tnithin236/stock-market-ai

Suggested hashtags: #MachineLearning #TimeSeries #PyTorch #XGBoost #FastAPI #MLOps #DataScience

### Short post version

I spent weeks building an ML system to predict stock moves - and the most valuable result was learning how to prove it
doesn't work. Leakage-safe pipeline, XGBoost/LSTM/Transformer, confidence intervals, significance tests, cost-aware
backtests: no statistically significant edge, and buy-and-hold won after costs. Code, tests and write-up on GitHub.
Rigour over hype. (Educational project, not financial advice.)
