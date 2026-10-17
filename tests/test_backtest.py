import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import MarketData, build_market_data, net_returns, run_backtest, trade_events
from src.backtesting.metrics import _trade_returns, max_drawdown, performance_metrics, sharpe_from_returns
from src.backtesting.random_baseline import null_summary, random_null
from src.backtesting.signals import buy_and_hold, positions_from_prob, positions_from_return
from src.evaluation.predictions import load_predictions, save_predictions
from src.features.engineering import build_features, make_supervised


def _market(overnight, intraday):
    overnight, intraday = np.asarray(overnight, float), np.asarray(intraday, float)
    n = len(overnight)
    dates = pd.bdate_range("2025-01-01", periods=n)
    return MarketData(signal_dates=dates, pnl_dates=dates + pd.offsets.BDay(1), close=np.full(n, 100.0),
                      overnight=overnight, intraday=intraday, ctc=(1 + overnight) * (1 + intraday) - 1)


ON = [0.01, -0.02, 0.005, 0.0, 0.015]
ID = [0.002, 0.01, -0.01, 0.004, -0.003]


def test_buy_and_hold_same_close_equals_compounded_close_to_close():
    m = _market(ON, ID)
    bt = run_backtest(m, np.ones(5), 1000.0, cost_bps=0.0, execution="same_close")
    assert np.isclose(bt["equity"].iloc[-1], 1000.0 * np.prod(1 + m.ctc))


def test_buy_and_hold_next_open_misses_only_the_first_overnight_gap():
    m = _market(ON, ID)
    bt = run_backtest(m, np.ones(5), 1000.0, cost_bps=0.0, execution="next_open")
    expected = 1000.0 * (1 + ID[0]) * np.prod([(1 + ON[i]) * (1 + ID[i]) for i in range(1, 5)])
    assert np.isclose(bt["equity"].iloc[-1], expected)


def test_flat_strategy_keeps_capital_and_pays_nothing():
    bt = run_backtest(_market(ON, ID), np.zeros(5), 1000.0, 25.0, "next_open")
    assert np.allclose(bt["equity"], 1000.0) and bt["turnover"].sum() == 0 and bt["cost_ret"].sum() == 0


def test_costs_are_charged_on_turnover():
    m = _market(ON, ID)
    pos = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    bt = run_backtest(m, pos, 1000.0, cost_bps=10.0, execution="same_close")
    assert np.isclose(bt["turnover"].sum(), 5.0)                     # every day the position changes by 1
    assert np.isclose(bt["cost_ret"].sum(), 5 * 0.001)
    assert np.isclose(bt["net_ret"].iloc[0], (1 - 0.001) * (1 + m.ctc[0]) - 1)
    assert np.isclose(bt["net_ret"].iloc[1], -0.001)                 # flat but paid to exit


def test_shorting_earns_the_negative_of_the_move():
    m = _market(ON, ID)
    net, gross, *_ = net_returns(-np.ones(5), m, 0.0, "same_close")
    assert np.allclose(gross[0], -m.ctc)


@pytest.mark.parametrize("execution", ["next_open", "same_close"])
def test_changing_future_positions_never_changes_past_returns(execution):
    m = _market(ON * 4, ID * 4)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 20).astype(float)
    b = a.copy()
    k = 12
    b[k:] = 1.0 - b[k:]
    net_a, *_ = net_returns(a, m, 5.0, execution)
    net_b, *_ = net_returns(b, m, 5.0, execution)
    assert np.allclose(net_a[0, :k], net_b[0, :k])                   # rows before k untouched


def test_max_drawdown_known_value():
    assert np.isclose(max_drawdown([100, 120, 90, 110]), 90 / 120 - 1)


def test_sharpe_zero_for_constant_returns_and_matches_formula():
    assert float(sharpe_from_returns(np.full(50, 0.001))) == 0.0
    r = np.random.default_rng(1).normal(0.001, 0.01, 200)
    assert np.isclose(sharpe_from_returns(r), r.mean() / r.std(ddof=1) * np.sqrt(252))


def test_trade_returns_split_on_flat_days_and_direction_changes():
    pos = np.array([0, 1, 1, 0, 1, -1, 0.0])
    net = np.array([0.0, 0.10, -0.05, 0.0, 0.02, 0.03, 0.0])
    tr = _trade_returns(pos, net)
    assert np.allclose(tr, [1.10 * 0.95 - 1, 0.02, 0.03])


def test_performance_metrics_are_consistent():
    m = _market(ON * 6, ID * 6)
    pos = np.tile([1.0, 1.0, 0.0, 1.0, 0.0], 6)
    bt = run_backtest(m, pos, 5000.0, 5.0, "next_open")
    pm = performance_metrics(bt, 5000.0)
    assert np.isclose(pm["total_return_%"], (bt["equity"].iloc[-1] / 5000 - 1) * 100)
    assert np.isclose(pm["exposure_%"], pos.mean() * 100)
    assert pm["max_drawdown_%"] <= 0 and 0 <= pm["trade_win_rate_%"] <= 100
    assert np.isclose(pm["turnover"], bt["turnover"].sum())


def test_market_data_matches_next_day_prices(ohlcv):
    full = build_features(ohlcv)
    sup = make_supervised(full)
    idx = sup.index[-50:]
    m = build_market_data(full, idx)
    assert np.allclose(m.ctc, sup.loc[idx, "target_return"])           # close(t) -> close(t+1)
    assert np.allclose((1 + m.overnight) * (1 + m.intraday) - 1, m.ctc)
    assert (m.pnl_dates > m.signal_dates).all()
    pos = full.index.get_indexer(idx)
    assert (m.pnl_dates == full.index[pos + 1]).all()                  # really the NEXT trading day


def test_random_null_matches_requested_exposure_and_buy_and_hold_is_never_beaten_at_full_exposure():
    m = _market(ON * 10, ID * 10)
    sh, tot = random_null(m, 0.5, 400, 5.0, "next_open", False, seed=1)
    assert sh.shape == tot.shape == (400,)
    sh1, _ = random_null(m, 1.0, 50, 0.0, "same_close", False, seed=1)
    bh = float(sharpe_from_returns(run_backtest(m, np.ones(50), 1.0, 0.0, "same_close")["net_ret"]))
    assert np.allclose(sh1, bh)
    s = null_summary(bh, sh1)
    assert s["p_value"] == 1.0 and s["beats_random_%"] == 0.0


def test_signal_helpers():
    p = np.array([0.4, 0.5, 0.6, 0.9])
    assert list(positions_from_prob(p)) == [0, 0, 1, 1]
    assert list(positions_from_prob(p, allow_short=True)) == [-1, -1, 1, 1]
    assert np.allclose(positions_from_prob(p, sizing="scaled", scale=5), [0, 0, 0.5, 1.0])
    assert list(positions_from_return([-0.01, 0.0, 0.02])) == [0, 0, 1]
    assert list(buy_and_hold(3)) == [1, 1, 1]
    with pytest.raises(ValueError):
        positions_from_prob(p, sizing="kelly")


def test_trade_events_mark_changes_only():
    m = _market(ON, ID)
    ev = trade_events(m, np.array([0, 1, 1, 0, 1.0]))
    assert list(ev["action"]) == ["BUY", "SELL", "BUY"]


def test_prediction_files_roundtrip_and_filter_by_split():
    idx = pd.bdate_range("2025-01-01", periods=4, name="Date")
    f1 = pd.DataFrame({"split": "val", "pred_return": 0.0, "raw_prob": 0.5, "cal_prob": 0.5}, index=idx[:2])
    f2 = pd.DataFrame({"split": "test", "pred_return": 0.1, "raw_prob": 0.6, "cal_prob": 0.55}, index=idx[2:])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.csv"
        save_predictions(path, [f1, f2])
        test = load_predictions(path, "test")
    assert list(test.index) == list(idx[2:]) and np.allclose(test["raw_prob"], 0.6)
