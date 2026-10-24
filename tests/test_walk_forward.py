import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from src.data.synthetic import make_ohlcv  # noqa: E402
from src.features.engineering import build_features, feature_columns, make_supervised  # noqa: E402
from src.training.walk_forward import (fold_sign_test, make_folds, walk_forward_predictions,  # noqa: E402
                                       yearly_table)

PARAMS = {"max_depth": 2, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}


def _run(sup, **kw):
    cols = feature_columns(sup, scale_free_only=True)
    args = dict(params=PARAMS, n_estimators=25, early_stopping_rounds=5, seed=0, use_class_weight=True,
                initial_train=400, test_size=100, gap=1)
    args.update(kw)
    return walk_forward_predictions(sup, cols, **args)


def test_folds_are_expanding_ordered_and_leak_free():
    folds = make_folds(1000, initial_train=400, test_size=100, gap=1)
    assert len(folds) >= 5
    for f in folds:
        assert f["test_start"] == f["train_end"] + 1               # gap rows between training and test
        assert f["test_end"] - f["test_start"] <= 100 and f["test_end"] <= 1000
    for a, b in zip(folds, folds[1:]):
        assert b["train_end"] == a["test_end"]                      # training window expands, blocks do not overlap
        assert b["test_start"] > a["test_end"]


def test_invalid_fold_settings_raise():
    with pytest.raises(ValueError):
        make_folds(1000, 10, 100)
    with pytest.raises(ValueError):
        make_folds(1000, 400, 5)


def test_predictions_cover_each_test_day_once_in_time_order():
    sup = make_supervised(build_features(make_ohlcv(1200)))
    preds, folds = _run(sup)
    assert preds.index.is_unique and preds.index.is_monotonic_increasing
    assert len(preds) == int(folds["n_test"].sum())
    assert preds["raw_prob"].between(0, 1).all()
    assert preds.index.min() == sup.index[401]                       # first test row = initial_train + gap


def test_test_labels_never_influence_their_own_fold():
    sup = make_supervised(build_features(make_ohlcv(1200)))
    base, _ = _run(sup)
    f0 = make_folds(len(sup), 400, 100, 1)[0]
    tampered = sup.copy()
    rows = tampered.index[f0["test_start"]:f0["test_end"]]
    tampered.loc[rows, "target_direction"] = 1 - tampered.loc[rows, "target_direction"]
    tampered.loc[rows, "target_return"] = -tampered.loc[rows, "target_return"]
    changed, _ = _run(tampered)
    pd.testing.assert_frame_equal(base[base["fold"] == 0], changed[changed["fold"] == 0])


def test_too_little_data_raises():
    sup = make_supervised(build_features(make_ohlcv(300)))
    with pytest.raises(ValueError):
        _run(sup, initial_train=5000)


def test_fold_sign_test_and_yearly_table():
    s = fold_sign_test(pd.Series([0.55, 0.52, 0.51, 0.49, 0.56, 0.53, 0.54, 0.58, 0.52, 0.5]))
    assert s["folds"] == 10 and s["folds_auc_above_0.5"] == 8 and 0 < s["p_value"] < 1
    idx = pd.bdate_range("2023-11-01", periods=120)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"target_direction": rng.integers(0, 2, 120), "raw_prob": rng.uniform(size=120)}, index=idx)
    years = yearly_table(frame)
    assert list(years.index) == [2023, 2024] and years["days"].sum() == 120
