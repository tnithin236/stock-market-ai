import tempfile
from pathlib import Path

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from src.data.splits import chronological_split  # noqa: E402
from src.features.engineering import build_features, feature_columns, make_supervised  # noqa: E402
from src.models.xgb_models import (PlattCalibrator, fit_model, load_bundle, make_model,  # noqa: E402
                                   predict_proba_up, predict_return, predict_with_bundle,
                                   return_to_price, save_bundle)

PARAMS = {"max_depth": 2, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}


def _split(ohlcv):
    sup = make_supervised(build_features(ohlcv))
    return sup, chronological_split(sup, 0.7, 0.15, gap=1)


def test_regressor_predicts_returns_and_reconstructs_price(ohlcv):
    sup, s = _split(ohlcv)
    cols = feature_columns(sup, scale_free_only=True)
    m = make_model("regression", PARAMS, n_estimators=40, early_stopping_rounds=5)
    fit_model(m, s.train[cols], s.train["target_return"], s.val[cols], s.val["target_return"])
    pr = predict_return(m, s.test[cols])
    assert pr.shape == (len(s.test),) and np.isfinite(pr).all()
    price = return_to_price(s.test["Close"], pr)
    assert np.allclose(price, s.test["Close"].to_numpy() * (1 + pr))


def test_classifier_probabilities_are_valid(ohlcv):
    sup, s = _split(ohlcv)
    cols = feature_columns(sup)
    m = make_model("classification", PARAMS, n_estimators=40, early_stopping_rounds=5, scale_pos_weight=0.9)
    fit_model(m, s.train[cols], s.train["target_direction"], s.val[cols], s.val["target_direction"])
    p = predict_proba_up(m, s.test[cols])
    assert p.shape == (len(s.test),) and ((p >= 0) & (p <= 1)).all()


def test_platt_calibrator_is_monotonic_and_bounded():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.2, 0.8, 300)
    y = (rng.uniform(size=300) < raw).astype(int)
    cal = PlattCalibrator().fit(raw, y)
    grid = np.linspace(0.05, 0.95, 50)
    out = cal.predict(grid)
    assert ((out >= 0) & (out <= 1)).all()
    assert np.all(np.diff(out) >= -1e-12)


def test_bundle_roundtrip_gives_identical_predictions(ohlcv):
    sup, s = _split(ohlcv)
    cols = feature_columns(sup)
    m = make_model("classification", PARAMS, n_estimators=30, early_stopping_rounds=5)
    fit_model(m, s.train[cols], s.train["target_direction"], s.val[cols], s.val["target_direction"])
    cal = PlattCalibrator().fit(predict_proba_up(m, s.val[cols]), s.val["target_direction"])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bundle.joblib"
        save_bundle(path, m, "classification", cols, cal, {"note": "test"})
        bundle = load_bundle(path)
    a = predict_with_bundle({"model": m, "task": "classification", "features": cols, "calibrator": cal}, s.test)
    b = predict_with_bundle(bundle, s.test)
    assert np.allclose(a["raw_score"], b["raw_score"])
    assert np.allclose(a["calibrated_prob"], b["calibrated_prob"])


def test_tuning_returns_valid_params(ohlcv):
    pytest.importorskip("optuna")
    from src.training.tuning import tune_xgboost

    sup, s = _split(ohlcv)
    cols = feature_columns(sup, scale_free_only=True)
    params, value, n_run = tune_xgboost(
        "regression", s.train[cols], s.train["target_return"], s.val[cols], s.val["target_return"],
        n_trials=2, seed=0, n_estimators=30, early_stopping_rounds=5)
    assert n_run == 2 and np.isfinite(value)
    assert {"max_depth", "learning_rate", "subsample", "colsample_bytree"} <= set(params)
    assert 2 <= params["max_depth"] <= 6


def test_tree_contributions_sum_to_raw_margin(ohlcv):
    if not hasattr(xgb, "DMatrix"):
        pytest.skip("real xgboost required")
    from src.evaluation.explain import tree_contributions

    sup, s = _split(ohlcv)
    cols = feature_columns(sup, scale_free_only=True)
    m = make_model("regression", PARAMS, n_estimators=40, early_stopping_rounds=5)
    fit_model(m, s.train[cols], s.train["target_return"], s.val[cols], s.val["target_return"])
    contrib, bias = tree_contributions(m, s.test[cols])
    margin = m.predict(s.test[cols], iteration_range=(0, int(m.best_iteration) + 1))
    assert np.allclose(contrib.sum(axis=1).to_numpy() + bias, margin, atol=1e-4)
