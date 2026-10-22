import tempfile
from pathlib import Path

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import api.main as api_main  # noqa: E402
from tests.serving_helpers import build_service  # noqa: E402


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as d:
        api_main.service = build_service(Path(d))
        yield TestClient(api_main.app)
        api_main.service = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["trained_tickers"] == ["AAPL"]


def test_stock_endpoint(client):
    r = client.get("/stock/aapl?days=7")
    assert r.status_code == 200 and len(r.json()["rows"]) == 7
    assert client.get("/stock/AAPL?days=1").status_code == 422          # days must be >= 2
    assert client.get("/stock/AA PL").status_code == 422


def test_predict_endpoint_shape_and_honesty(client):
    r = client.post("/predict", json={"ticker": "AAPL"})
    body = r.json()
    assert r.status_code == 200 and body["prediction"] in ("UP", "DOWN")
    assert body["score_is_probability"] is False and "confidence" not in body
    assert any("not financial advice" in w.lower() for w in body["warnings"])


def test_predict_errors(client):
    assert client.post("/predict", json={"ticker": "MSFT"}).status_code == 404
    assert client.post("/predict", json={"ticker": "AA PL"}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_metrics_endpoint(client):
    r = client.get("/metrics/AAPL")
    assert r.status_code == 200 and r.json()["interpretation"].startswith("No statistically significant")
    assert client.get("/metrics/MSFT").status_code == 404


def test_train_runs_in_background_and_reports_status(client):
    r = client.post("/train", json={"ticker": "AAPL", "start_date": "2015-01-01"})
    assert r.status_code == 202 and r.json()["status"] == "queued"
    job_id = r.json()["job_id"]
    s = client.get(f"/train/{job_id}")                                   # the TestClient has already run the task
    assert s.status_code == 200 and s.json()["status"] == "succeeded"
    assert client.get("/train/unknown").status_code == 404
    assert client.post("/train", json={"ticker": "AAPL", "start_date": "bad"}).status_code == 422


def test_predict_model_parameter_and_health_models(client):
    assert client.get("/health").json()["models"] == {"AAPL": ["xgboost"]}
    assert client.post("/predict", json={"ticker": "AAPL", "model": "xgboost"}).status_code == 200
    assert client.post("/predict", json={"ticker": "AAPL", "model": "lstm"}).status_code == 404
    assert client.post("/predict", json={"ticker": "AAPL", "model": "gru"}).status_code == 422
    assert client.get("/metrics/AAPL?model=gru").status_code == 422
