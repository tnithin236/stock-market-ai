import numpy as np
import pytest

from src.data.splits import chronological_split
from src.features.engineering import build_features, feature_columns, make_supervised
from src.features.preprocessing import FeatureScaler


def _sup(ohlcv):
    return make_supervised(build_features(ohlcv))


def test_split_is_chronological_and_non_overlapping(ohlcv):
    s = chronological_split(_sup(ohlcv), 0.7, 0.15, gap=1)
    assert s.train.index.max() < s.val.index.min()
    assert s.val.index.max() < s.test.index.min()
    assert len(set(s.train.index) & set(s.val.index)) == 0


def test_split_sizes_roughly_match_fractions(ohlcv):
    sup = _sup(ohlcv)
    s = chronological_split(sup, 0.7, 0.15, gap=1)
    assert abs(len(s.train) / len(sup) - 0.70) < 0.01
    assert abs(len(s.val) / len(sup) - 0.15) < 0.02
    assert abs(len(s.test) / len(sup) - 0.15) < 0.02


def test_gap_removes_boundary_rows(ohlcv):
    sup = _sup(ohlcv)
    g0 = chronological_split(sup, 0.7, 0.15, gap=0)
    g1 = chronological_split(sup, 0.7, 0.15, gap=1)
    assert len(g1.val) == len(g0.val) - 1
    assert g1.val.index.min() > g0.val.index.min()


def test_bad_fractions_raise(ohlcv):
    with pytest.raises(ValueError):
        chronological_split(_sup(ohlcv), 0.9, 0.2)


def test_unsorted_index_raises(ohlcv):
    with pytest.raises(ValueError):
        chronological_split(_sup(ohlcv).iloc[::-1])


def test_scaler_is_fitted_on_train_only(ohlcv):
    sup = _sup(ohlcv)
    s = chronological_split(sup, 0.7, 0.15)
    cols = feature_columns(sup)
    scaler = FeatureScaler("standard").fit(s.train, cols)
    assert np.allclose(scaler.scaler.mean_, s.train[cols].mean().to_numpy())
    assert abs(scaler.transform(s.train).mean()).max() < 1e-8
    manual = (s.val[cols] - s.train[cols].mean()) / s.train[cols].std(ddof=0)
    assert np.allclose(scaler.transform(s.val).to_numpy(), manual.to_numpy())


def test_scaler_requires_fit(ohlcv):
    with pytest.raises(RuntimeError):
        FeatureScaler().transform(_sup(ohlcv))


def test_scaler_save_load_roundtrip(ohlcv, tmp_path_factory=None):
    import tempfile
    from pathlib import Path
    sup = _sup(ohlcv)
    cols = feature_columns(sup)
    scaler = FeatureScaler("robust").fit(sup.iloc[:300], cols)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "scaler.joblib"
        scaler.save(path)
        loaded = FeatureScaler.load(path)
    assert np.allclose(scaler.transform(sup).to_numpy(), loaded.transform(sup).to_numpy())
