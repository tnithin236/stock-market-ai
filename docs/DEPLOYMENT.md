# Deployment guide

Experimental research/educational system - not financial advice.

## 1. Local (no Docker)

    python -m venv .venv
    .venv\Scripts\Activate.ps1            # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    python main.py xgboost                # train + evaluate, saves models/ and reports/
    python main.py lstm                   # optional
    python main.py transformer            # optional
    python main.py backtest               # needs the predictions saved by the commands above
    uvicorn api.main:app --reload         # API  -> http://127.0.0.1:8000/docs
    streamlit run app/streamlit_app.py    # UI   -> http://localhost:8501

Trained artifacts: `models/<TICKER>_xgb_*.joblib`, `models/<TICKER>_lstm_*.pt|joblib`, `reports/*.json|csv`.
They are git-ignored; regenerate them with the commands above.

## 2. Docker

Build and run the API only:

    docker build -t stock-market-ai .
    docker run --rm -p 8000:8000 -v ${PWD}/models:/app/models -v ${PWD}/reports:/app/reports -v ${PWD}/data:/app/data stock-market-ai

API and dashboard together (the dashboard waits until the API is healthy):

    docker compose up --build

First run with no trained models (trains XGBoost for AAPL inside the container; needs internet):

    $env:AUTO_TRAIN = "1"; docker compose up --build        # bash: AUTO_TRAIN=1 docker compose up --build

Stop: `docker compose down`.

| Variable | Default | Meaning |
|---|---|---|
| `AUTO_TRAIN` | 0 | 1 = train an XGBoost model at start-up if the ticker has none |
| `AUTO_TRAIN_TICKER` | AAPL | Ticker for AUTO_TRAIN |
| `PORT` | 8000 | API port (honoured by the default container command; used by hosts such as Render) |
| `STOCKAI_SYNTHETIC` | 0 | 1 = dashboard uses fake market data (demos, offline) |

The image uses CPU-only PyTorch, runs as a non-root user, and excludes tests, notebooks, data and models
(mounted as volumes instead).

## 3. Cloud options

| Platform | Fit | Notes |
|---|---|---|
| Hugging Face Spaces (Docker space) | Dashboard | Free CPU; set the app port to 8501 and use the dashboard command from docker-compose.yml |
| Render / Railway / Fly.io (Docker) | API | Connect the GitHub repo, choose the Dockerfile; `PORT` is honoured. Free tiers have small RAM (PyTorch + XGBoost can exceed 512 MB) and sleep when idle |
| Streamlit Community Cloud | Dashboard | Installs from requirements.txt (includes the large default PyTorch build); create a slimmer requirements file for it |

Cloud caveats:
* The disk is ephemeral. Models are not in git, so set `AUTO_TRAIN=1` (trains XGBoost on first start, about a minute) or bake
  artifacts into your own image.
* Training and Yahoo Finance downloads need outbound internet access; Yahoo can rate-limit shared cloud IPs.
* Only XGBoost is trained automatically. Run `python main.py lstm|transformer` in a one-off job if you want those too.
* Add authentication and rate limiting before exposing the API publicly. Do not present predictions as advice.

## 4. CI

`.github/workflows/ci.yml` runs the test suite and builds the Docker image on every push and pull request.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `docker: command not found` | Install Docker Desktop and start it |
| Dashboard container never starts | It waits for the API healthcheck; check `docker compose logs api` |
| API unhealthy on first run | Training takes about a minute with AUTO_TRAIN=1; the healthcheck allows 120 s plus retries |
| Permission denied writing models (Linux host) | `chmod -R 777 models reports data` or run the container with your uid |
| `No trained model` | Run training once, or set AUTO_TRAIN=1 |
