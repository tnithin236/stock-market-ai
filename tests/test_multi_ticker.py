import numpy as np
import pytest

pytest.importorskip("xgboost")

import src.training.multi_ticker as mt  # noqa: E402
from src.data.synthetic import make_ohlcv  # noqa: E402
from src.features.engineering import build_features, make_supervised  # noqa: E402

CFG = {"walk_forward": {"initial_train_days": 400, "test_days": 100}, "backtest": {},
       "xgboost": {"n_estimators": 30, "early_stopping_rounds": 5, "params": {"max_depth": 2}},
       "multi_ticker": {"tickers": ["AAA", "BBB", "CCC"], "alpha": 0.05}}


def test_holm_and_benjamini_hochberg_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(mt.holm_bonferroni(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(mt.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


def test_adjusted_p_values_are_ordered_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=40)
    holm, bh = mt.holm_bonferroni(p), mt.benjamini_hochberg(p)
    assert (holm >= p - 1e-12).all() and (bh >= p - 1e-12).all()
    assert (holm <= 1).all() and (bh <= 1).all()
    assert (holm >= bh - 1e-12).all()                        # Holm is the more conservative of the two


def test_auc_pvalue_detects_signal_and_not_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    assert mt.auc_pvalue(y, y + rng.normal(0, 0.5, 500)) < 1e-6
    assert mt.auc_pvalue(y, rng.normal(size=500)) > 0.01
    assert np.isnan(mt.auc_pvalue(np.ones(10), rng.normal(size=10)))


def _fake_loader(cfg, ticker, *a, **k):
    if ticker == "BAD":
        raise RuntimeError("no data for BAD")
    full = build_features(make_ohlcv(1200, seed=sum(map(ord, ticker))))
    return ticker, full, make_supervised(full)


def _with_fake(fn):
    original = mt.load_supervised
    mt.load_supervised = _fake_loader
    try:
        return fn()
    finally:
        mt.load_supervised = original


def test_study_runs_reports_summary_and_survives_a_failing_ticker():
    out = _with_fake(lambda: mt.run_multi_ticker(CFG, ["AAA", "BBB", "CCC", "BAD"], make_plots=False, save=False))
    assert out["tickers"] == ["AAA", "BBB", "CCC"] and list(out["failed"]) == ["BAD"]
    s = out["summary"]
    assert 0 <= s["n_auc_gt_half"] <= 3 and 0 <= s["holm_significant"] <= s["nominal_significant"] <= 3
    assert abs(s["expected_by_chance"] - 0.15) < 1e-9
    row = out["per_ticker"][0]
    for key in ("auc", "auc_lo", "auc_hi", "auc_p", "auc_p_holm", "clf_sharpe", "bh_sharpe", "rmse_vs_naive"):
        assert key in row
    assert 0 <= row["auc"] <= 1 and row["auc_lo"] <= row["auc"] <= row["auc_hi"]


def test_study_needs_at_least_two_tickers():
    with pytest.raises(ValueError):
        _with_fake(lambda: mt.run_multi_ticker(CFG, ["AAA", "BAD"], make_plots=False, save=False))
