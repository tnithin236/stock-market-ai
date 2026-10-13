import numpy as np

from src.evaluation.metrics import classification_metrics, directional_accuracy, regression_metrics
from src.features.engineering import build_features, make_supervised
from src.models.baselines import (MajorityClassBaseline, MeanReversionBaseline,
                                  MomentumBaseline, MovingAverageBaseline, PreviousDayBaseline)


def test_regression_metrics_perfect():
    m = regression_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["MAPE_%"] == 0 and m["R2"] == 1


def test_regression_metrics_known_values():
    m = regression_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 4.0])
    assert np.isclose(m["MAE"], 2 / 3)
    assert np.isclose(m["RMSE"], np.sqrt(2 / 3))
    assert np.isclose(m["MAPE_%"], (1 + 0 + 1 / 3) / 3 * 100)


def test_classification_metrics_known_values():
    m = classification_metrics([1, 0, 1, 1], [1, 0, 0, 1], y_score=[0.9, 0.2, 0.4, 0.8])
    assert m["accuracy"] == 0.75
    assert m["precision"] == 1.0
    assert np.isclose(m["recall"], 2 / 3)
    assert m["confusion_matrix"] == [[1, 0], [1, 2]]
    assert m["roc_auc"] == 1.0


def test_roc_auc_none_when_single_class():
    m = classification_metrics([1, 1, 1], [1, 1, 1], y_score=[0.5, 0.6, 0.7])
    assert m["roc_auc"] is None


def test_directional_accuracy():
    prev = np.array([10.0, 10.0, 10.0, 10.0])
    true = np.array([11.0, 9.0, 11.0, 9.0])
    pred = np.array([12.0, 8.0, 9.0, 12.0])
    assert directional_accuracy(prev, true, pred) == 0.5
    assert np.isnan(directional_accuracy(prev, true, prev))


def test_regression_baselines(ohlcv):
    sup = make_supervised(build_features(ohlcv))
    assert np.array_equal(PreviousDayBaseline().predict(sup), sup["Close"].to_numpy())
    assert np.array_equal(MovingAverageBaseline(20).predict(sup), sup["sma_20"].to_numpy())


def test_classification_baselines(ohlcv):
    sup = make_supervised(build_features(ohlcv))
    maj = MajorityClassBaseline().fit(sup["target_direction"])
    assert maj.label == int(sup["target_direction"].mean() >= 0.5)
    assert set(maj.predict(sup)) == {maj.label}
    mom, rev = MomentumBaseline().predict(sup), MeanReversionBaseline().predict(sup)
    nonzero = sup["return"].to_numpy() != 0
    assert (mom[nonzero] + rev[nonzero] == 1).all()
