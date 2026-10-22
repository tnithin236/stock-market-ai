# API reference

Start the server (from the project root, venv active):

    uvicorn api.main:app --reload

Interactive docs: http://127.0.0.1:8000/docs

| Method | Path | Purpose |
|---|---|---|
| GET | /health | Liveness, version, tickers with trained models |
| GET | /stock/{ticker}?days=30 | Recent OHLCV bars, period return, annualized volatility |
| POST | /predict | Next-session UP/DOWN and predicted close. Body: `{"ticker":"AAPL","model":"xgboost"}` (`model` = xgboost, lstm or transformer; default xgboost) |
| POST | /train | Start background training (returns 202 + job id) |
| GET | /train/{job_id} | Poll a training job |
| GET | /metrics/{ticker}?model=xgboost | Saved test-period metrics and backtest summary for a model |

## /predict example

    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"ticker":"AAPL"}'

Key response fields: `prediction`, `predicted_close`, `score` (raw model score, **not** a probability),
`score_is_probability` (always false), `calibrated_probability_up` (Platt-calibrated on validation; with a weak
signal it stays near the historical share of up days), `evidence` (held-out test accuracy / AUC and whether
the edge is statistically significant), `warnings`, `disclaimer`.

`/health` also lists which models exist per ticker.

Errors: 404 no trained model, 409 training already running, 422 invalid input or too little history,
502 market data unavailable.

Experimental research/educational system. Not financial advice.
