"""Request/response models for the HTTP API (pydantic v2)."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    trained_tickers: List[str]
    models: Dict[str, List[str]] = {}
    time_utc: str


class Bar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class StockResponse(BaseModel):
    ticker: str
    rows: List[Bar]
    latest_close: float
    period_return_pct: float
    annualized_volatility_pct: Optional[float] = None


class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=15, examples=["AAPL"])
    model: Literal["xgboost", "lstm", "transformer"] = "xgboost"


class PredictResponse(BaseModel):
    ticker: str
    as_of_date: str
    prediction: Literal["UP", "DOWN"]
    predicted_close: float
    predicted_return_pct: float
    last_close: float
    score: float = Field(..., description="Raw model score. NOT a probability.")
    score_is_probability: bool = False
    calibrated_probability_up: Optional[float] = None
    calibration_note: str
    evidence: Optional[Dict[str, Any]] = None
    details: Dict[str, Any]
    warnings: List[str]
    disclaimer: str


class TrainRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=15, examples=["AAPL"])
    start_date: Optional[str] = Field(None, examples=["2015-01-01"])
    end_date: Optional[str] = Field(None, description="Exclusive end date; omit for the latest data.")


class TrainResponse(BaseModel):
    job_id: str
    ticker: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    ticker: str
    status: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class MetricsResponse(BaseModel):
    ticker: str
    test_period: Optional[List[str]] = None
    regression: List[Dict[str, Any]]
    classification: List[Dict[str, Any]]
    backtest: Optional[Dict[str, Any]] = None
    interpretation: str
    disclaimer: str
