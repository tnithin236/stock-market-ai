import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dashboard.data import add_indicators, get_market_data, summary_stats
from src.dashboard.results import (direction_summary, load_results, permutation_table, prediction_frame,
                                   results_tables)
from src.data.synthetic import make_ohlcv
from src.evaluation.predictions import save_predictions


def test_indicator_columns_and_ranges():
    out = add_indicators(make_ohlcv(300))
    for c in ["sma_20", "sma_50", "bb_upper", "bb_lower", "rsi_14", "macd", "macd_signal", "macd_hist"]:
        assert c in out.columns
    assert out["rsi_14"].dropna().between(0, 100).all()
    ok = out.dropna(subset=["bb_upper"])
    assert (ok["bb_upper"] >= ok["bb_lower"]).all()


def test_indicators_are_causal():
    df = make_ohlcv(300)
    a, b = add_indicators(df.iloc[:200]), add_indicators(df)
    pd.testing.assert_frame_equal(a, b.iloc[:200], check_exact=False, rtol=1e-9)


def test_summary_stats():
    s = summary_stats(make_ohlcv(300))
    assert s["n_days"] == 300 and s["max_drawdown_pct"] <= 0 and s["annualized_volatility_pct"] > 0
    assert np.isclose(s["period_return_pct"], (make_ohlcv(300)["Close"].iloc[-1] / make_ohlcv(300)["Close"].iloc[0] - 1) * 100)


def test_synthetic_mode_and_ticker_validation(monkeypatch):
    monkeypatch.setenv("STOCKAI_SYNTHETIC", "1")
    df = get_market_data("aapl", "2021-01-01", "2022-01-01", {})
    assert df.index.min() >= pd.Timestamp("2021-01-01") and df.index.max() <= pd.Timestamp("2022-01-01")
    assert len(get_market_data("AAPL", "1990-01-01", "1990-02-01", {})) > 0        # empty slice -> full fallback
    with pytest.raises(ValueError):
        get_market_data("AA PL", "2021-01-01", "2022-01-01", {})


def _write_artifacts(d: Path, df: pd.DataFrame):
    idx = df.index[-60:-1]
    frame = pd.DataFrame({"split": "test", "pred_return": 0.001, "raw_prob": np.linspace(0.4, 0.6, len(idx)),
                          "cal_prob": 0.53}, index=idx)
    save_predictions(d / "predictions_lstm_AAPL.csv", [frame])
    res = {"test_period": ["a", "b"],
           "regression": {"test": [{"model": "previous_day", "RMSE": 1.0}, {"model": "lstm_reg", "RMSE": 1.1}]},
           "classification": {"test": [{"model": "always_majority", "accuracy": 0.5}]},
           "permutation_importance": {"classifier": [{"feature": "rsi_14", "importance_mean": 0.01, "importance_std": 0.01}]}}
    (d / "lstm_results_AAPL.json").write_text(json.dumps(res))


def test_results_and_prediction_frames():
    df = make_ohlcv(300)
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        assert load_results(d, "AAPL", "lstm") is None and prediction_frame(d, "AAPL", "lstm", df) is None
        _write_artifacts(d, df)
        res = load_results(d, "AAPL", "lstm")
        reg, cls = results_tables(res)
        frame = prediction_frame(d, "AAPL", "lstm", df)
        perm = permutation_table(res)
        assert permutation_table(res, "regressor") is None
    assert list(reg.index) == ["previous_day", "lstm_reg"] and list(cls.index) == ["always_majority"]
    assert list(perm.index) == ["rsi_14"]
    assert len(frame) == 59 and np.allclose(frame["model_pred_close"], frame["close"] * 1.001)
    assert np.allclose(frame["actual_next_close"], df["Close"].shift(-1).reindex(frame.index))
    s = direction_summary(frame)
    assert s["n_days"] == 59 and 0 <= s["accuracy"] <= 1 and 0 <= s["always_up_accuracy"] <= 1


def test_prediction_frame_drops_days_outside_the_loaded_range():
    df = make_ohlcv(300)
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        _write_artifacts(d, df)
        frame = prediction_frame(d, "AAPL", "lstm", df.iloc[:-30])          # user selected a shorter range
    assert frame is not None and frame.index.max() < df.index[-30]


def test_xgb_importance_returns_gain_series():
    pytest.importorskip("xgboost")
    from src.dashboard.explain_view import xgb_importance
    from src.features.engineering import build_features
    from tests.serving_helpers import build_service

    with tempfile.TemporaryDirectory() as t:
        build_service(Path(t))
        out = xgb_importance(Path(t) / "models", "AAPL", build_features(make_ohlcv(900)))
        missing = xgb_importance(Path(t) / "models", "MSFT", build_features(make_ohlcv(900)))
    assert missing is None
    assert out["gain"].index.is_unique and out["gain"].sum() == pytest.approx(1.0, abs=1e-6) or out["gain"].sum() == 0
