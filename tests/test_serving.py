import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")

from src.serving.errors import (DataUnavailableError, InsufficientHistoryError, ModelNotTrainedError,  # noqa: E402
                                TrainingInProgressError)
from src.serving.jobs import JobRegistry  # noqa: E402
from src.serving.service import clean  # noqa: E402
from tests.helpers import make_ohlcv  # noqa: E402
from tests.serving_helpers import build_service  # noqa: E402


def test_predict_is_consistent_and_never_calls_the_score_a_probability():
    with tempfile.TemporaryDirectory() as d:
        out = build_service(Path(d)).predict("aapl")
    assert out["ticker"] == "AAPL" and out["prediction"] in ("UP", "DOWN")
    assert (out["prediction"] == "UP") == (out["score"] > 0.5)
    assert np.isclose(out["predicted_close"], out["last_close"] * (1 + out["predicted_return_pct"] / 100))
    assert out["score_is_probability"] is False
    assert 0.0 <= out["calibrated_probability_up"] <= 1.0
    assert "confidence" not in out
    assert out["details"]["model"] == "xgboost" and out["details"]["prediction_horizon"] == "next trading session"
    json.dumps(out, allow_nan=False)                                     # JSON-safe


def test_predict_warns_about_no_edge_disclaimer_and_stale_data():
    with tempfile.TemporaryDirectory() as d:
        out = build_service(Path(d)).predict("AAPL")
    text = " ".join(out["warnings"]).lower()
    assert "not financial advice" in text and "no statistically significant" in text and "stale" in text
    assert out["evidence"]["significant_edge"] is False


def test_significant_evidence_removes_the_no_edge_warning():
    with tempfile.TemporaryDirectory() as d:
        out = build_service(Path(d), significant=True).predict("AAPL")
    assert out["evidence"]["significant_edge"] is True
    assert "no statistically significant" not in " ".join(out["warnings"]).lower()


def test_untrained_and_invalid_tickers():
    with tempfile.TemporaryDirectory() as d:
        svc = build_service(Path(d))
        with pytest.raises(ModelNotTrainedError):
            svc.predict("MSFT")
        for bad in ("AA PL", "", "../x", "X" * 30):
            with pytest.raises(ValueError):
                svc.predict(bad)


def test_short_history_and_data_failures_are_reported_cleanly():
    with tempfile.TemporaryDirectory() as d:
        svc = build_service(Path(d))
        svc.data_loader = lambda t: make_ohlcv(30)
        with pytest.raises(InsufficientHistoryError):
            svc.predict("AAPL")

        def boom(t):
            raise RuntimeError("network down")

        svc.data_loader = boom
        with pytest.raises(DataUnavailableError):
            svc.predict("AAPL")
        svc.data_loader = lambda t: pd.DataFrame()
        with pytest.raises(DataUnavailableError):
            svc.get_stock("AAPL")


def test_get_stock_returns_recent_bars():
    with tempfile.TemporaryDirectory() as d:
        out = build_service(Path(d)).get_stock("AAPL", days=10)
    assert len(out["rows"]) == 10 and set(out["rows"][0]) == {"date", "open", "high", "low", "close", "volume"}
    assert out["latest_close"] == out["rows"][-1]["close"]
    json.dumps(out, allow_nan=False)


def test_metrics_are_json_safe_and_interpretation_is_honest():
    with tempfile.TemporaryDirectory() as d:
        svc = build_service(Path(d))
        out = svc.metrics("AAPL")
        assert out["backtest"] is None
        pd.DataFrame({"total_return_%": [54.1, 40.6], "sharpe": [0.99, 0.94]},
                     index=["buy_and_hold", "xgboost_clf"]).to_csv(Path(d) / "reports" / "backtest_summary_AAPL.csv")
        out2 = svc.metrics("AAPL")
        with pytest.raises(ModelNotTrainedError):
            svc.metrics("MSFT")
    json.dumps(out, allow_nan=False)                                     # the NaN in the file became None
    assert out["regression"][0]["dir_acc_%"] is None
    assert out["interpretation"].startswith("No statistically significant")
    assert set(out2["backtest"]) == {"buy_and_hold", "xgboost_clf"}


def test_trained_tickers_requires_both_models():
    with tempfile.TemporaryDirectory() as d:
        svc = build_service(Path(d))
        assert svc.trained_tickers() == ["AAPL"] and svc.health()["status"] == "ok"
        (Path(d) / "models" / "AAPL_xgb_regression.joblib").unlink()
        assert svc.trained_tickers() == []


def test_clean_converts_nan_numpy_and_dates():
    out = clean({"a": float("nan"), "b": np.float32(1.5), "c": [np.int64(3), float("inf")],
                 "d": pd.Timestamp("2025-01-02"), "e": np.bool_(True)})
    assert out == {"a": None, "b": 1.5, "c": [3, None], "d": "2025-01-02 00:00:00", "e": True}
    json.dumps(out, allow_nan=False)


def test_job_registry_success_failure_and_duplicates():
    reg = JobRegistry()
    job = reg.create("AAPL")
    assert job["status"] == "queued"
    with pytest.raises(TrainingInProgressError):
        reg.create("AAPL")
    reg.create("MSFT")                                                   # other tickers are independent
    reg.run(job["job_id"], lambda: {"done": 1})
    done = reg.get(job["job_id"])
    assert done["status"] == "succeeded" and done["result"] == {"done": 1} and done["finished_at"]
    again = reg.create("AAPL")                                           # allowed once finished
    reg.run(again["job_id"], lambda: (_ for _ in ()).throw(ValueError("too little data")))
    failed = reg.get(again["job_id"])
    assert failed["status"] == "failed" and "too little data" in failed["error"]
    assert reg.get("nope") is None


def test_start_training_validates_dates_and_runs_the_trainer():
    calls = []
    with tempfile.TemporaryDirectory() as d:
        svc = build_service(Path(d), trainer=lambda t, s, e: calls.append((t, s, e)) or {"ok": True})
        with pytest.raises(ValueError):
            svc.start_training("AAPL", "not-a-date", None)
        job, runner = svc.start_training("aapl", None, "2026-01-01")
        assert job["ticker"] == "AAPL" and svc.jobs.get(job["job_id"])["status"] == "queued"
        with pytest.raises(TrainingInProgressError):
            svc.start_training("AAPL", None, None)
        runner()
        assert svc.jobs.get(job["job_id"])["status"] == "succeeded"
    assert calls == [("AAPL", "2015-01-01", "2026-01-01")]


def test_model_parameter_available_models_and_evidence():
    with tempfile.TemporaryDirectory() as d:
        svc = build_service(Path(d))
        assert svc.available_models("AAPL") == ["xgboost"] and svc.available_models("MSFT") == []
        assert svc.health()["models"] == {"AAPL": ["xgboost"]}
        assert svc.evidence("AAPL")["significant_edge"] is False and svc.evidence("AAPL", "lstm") is None
        with pytest.raises(ValueError):
            svc.predict("AAPL", model="gru")
        with pytest.raises(ModelNotTrainedError):
            svc.predict("AAPL", model="lstm")                           # no LSTM artifacts exist
        with pytest.raises(ValueError):
            svc.metrics("AAPL", model="gru")
        assert svc.predict("AAPL", model="xgboost")["details"]["model"] == "xgboost"
