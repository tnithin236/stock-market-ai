import numpy as np

from src.evaluation.stats import (accuracy_vs_rate_pvalue, bootstrap_auc_ci, r2_vs_zero,
                                  wilson_interval)


def test_wilson_contains_point_estimate_and_stays_in_unit_interval():
    lo, hi = wilson_interval(55, 100)
    assert 0 <= lo < 0.55 < hi <= 1


def test_wilson_narrows_with_more_data():
    w_small = np.subtract(*wilson_interval(55, 100)[::-1])
    w_large = np.subtract(*wilson_interval(550, 1000)[::-1])
    assert w_large < w_small


def test_pvalue_behaviour():
    assert accuracy_vs_rate_pvalue(70, 100, 0.5) < 0.001
    assert accuracy_vs_rate_pvalue(50, 100, 0.5) > 0.4


def test_bootstrap_auc_ci_brackets_auc():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    score = y * 0.5 + rng.normal(0, 1, 400)
    lo, hi = bootstrap_auc_ci(y, score, n_boot=300)
    assert lo <= roc_auc_score(y, score) <= hi
    assert 0.5 < lo


def test_r2_vs_zero():
    y = np.array([0.01, -0.02, 0.03, -0.01])
    assert r2_vs_zero(y, y) == 1.0
    assert r2_vs_zero(y, np.zeros(4)) == 0.0
    assert r2_vs_zero(y, -y) < 0


def test_platt_keeps_positive_slope_calibration_monotonic():
    from src.models.calibration import PlattCalibrator
    rng = np.random.default_rng(5)
    raw = rng.uniform(0.2, 0.8, 500)
    y = (rng.uniform(size=500) < raw).astype(int)
    cal = PlattCalibrator().fit(raw, y)
    out = cal.predict(np.linspace(0.1, 0.9, 30))
    assert cal.constant is None and np.all(np.diff(out) >= -1e-12)