"""FastAPI backend.   Run:  uvicorn api.main:app --reload   then open http://127.0.0.1:8000/docs

EXPERIMENTAL RESEARCH / EDUCATIONAL USE ONLY - NOT FINANCIAL ADVICE.
"""
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query

from api.schemas import (HealthResponse, JobStatus, MetricsResponse, PredictRequest, PredictResponse,
                         StockResponse, TrainRequest, TrainResponse)
from src.serving.errors import (DataUnavailableError, InsufficientHistoryError, ModelNotTrainedError,
                                TrainingInProgressError)
from src.serving.service import API_VERSION, PredictionService

app = FastAPI(
    title="Stock Market Prediction & Analysis API",
    version=API_VERSION,
    description=("Research/educational API around XGBoost models for next-day AAPL-style stock prediction. "
                 "Backtests found NO statistically significant predictive edge. Not financial advice."),
)

# Created once at import time; tests replace this attribute with a service wired to fake data.
service: Optional[PredictionService] = None


def _svc() -> PredictionService:
    global service
    if service is None:
        service = PredictionService.from_config()
    return service


def _translate(exc: Exception) -> HTTPException:
    if isinstance(exc, ModelNotTrainedError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, TrainingInProgressError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, DataUnavailableError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, (InsufficientHistoryError, ValueError)):
        return HTTPException(status_code=422, detail=str(exc))
    raise exc


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return _svc().health()


@app.get("/stock/{ticker}", response_model=StockResponse, tags=["data"])
def stock(ticker: str, days: int = Query(30, ge=2, le=1000, description="Most recent trading days to return")):
    try:
        return _svc().get_stock(ticker, days)
    except Exception as exc:  # noqa: BLE001
        raise _translate(exc)


@app.post("/predict", response_model=PredictResponse, tags=["model"])
def predict(req: PredictRequest):
    try:
        return _svc().predict(req.ticker, req.model)
    except Exception as exc:  # noqa: BLE001
        raise _translate(exc)


@app.post("/train", response_model=TrainResponse, status_code=202, tags=["model"])
def train(req: TrainRequest, background: BackgroundTasks):
    """Starts training in the background (typically under a minute). Poll GET /train/{job_id}."""
    try:
        job, runner = _svc().start_training(req.ticker, req.start_date, req.end_date)
    except Exception as exc:  # noqa: BLE001
        raise _translate(exc)
    background.add_task(runner)
    return {"job_id": job["job_id"], "ticker": job["ticker"], "status": job["status"],
            "message": f"Training started. Poll /train/{job['job_id']} for status."}


@app.get("/train/{job_id}", response_model=JobStatus, tags=["model"])
def train_status(job_id: str):
    job = _svc().jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return job


@app.get("/metrics/{ticker}", response_model=MetricsResponse, tags=["model"])
def metrics(ticker: str, model: str = Query("xgboost", pattern="^(xgboost|lstm|transformer)$")):
    try:
        return _svc().metrics(ticker, model)
    except Exception as exc:  # noqa: BLE001
        raise _translate(exc)
