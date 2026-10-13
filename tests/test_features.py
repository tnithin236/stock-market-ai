import numpy as np
import pandas as pd

from src.features.engineering import build_features, feature_columns, make_supervised


def test_expected_columns_present(ohlcv):
    f = build_features(ohlcv)
    for col in ["return", "log_return", "sma_10", "sma_20", "sma_50", "ema_10", "ema_20", "ema_50",
                "rsi_14", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower", "bb_pctb",
                "atr_14", "vol_10", "vol_20", "mom_5", "mom_10", "close_lag_1", "ret_lag_1",
                "volume_change", "target_close", "target_direction"]:
        assert col in f.columns, col


def test_rsi_is_bounded(ohlcv):
    rsi = build_features(ohlcv)["rsi_14"].dropna()
    assert rsi.between(0, 100).all()


def test_bollinger_bands_ordered(ohlcv):
    f = build_features(ohlcv).dropna(subset=["bb_upper"])
    assert (f["bb_upper"] >= f["bb_mid"]).all() and (f["bb_mid"] >= f["bb_lower"]).all()


def test_targets_are_next_day(ohlcv):
    f = build_features(ohlcv)
    assert np.allclose(f["target_close"].iloc[:-1], ohlcv["Close"].shift(-1).iloc[:-1])
    assert np.isnan(f["target_close"].iloc[-1])
    expected = (ohlcv["Close"].shift(-1) > ohlcv["Close"]).iloc[:-1].astype(float)
    assert (f["target_direction"].iloc[:-1] == expected).all()


def test_targets_never_in_feature_columns(ohlcv):
    cols = feature_columns(build_features(ohlcv))
    assert not any(c.startswith("target_") for c in cols)


def test_features_are_causal(ohlcv):
    """Features computed on a truncated history must equal the same rows of the full history."""
    cut = 400
    short = build_features(ohlcv.iloc[:cut])
    full = build_features(ohlcv)
    cols = feature_columns(full)
    pd.testing.assert_frame_equal(short[cols], full.iloc[:cut][cols], check_exact=False, rtol=1e-9)


def test_changing_the_future_does_not_change_the_past(ohlcv):
    altered = ohlcv.copy()
    altered.iloc[450:] = altered.iloc[450:] * 2.0
    a, b = build_features(ohlcv), build_features(altered)
    cols = feature_columns(a)
    pd.testing.assert_frame_equal(a.iloc[:450][cols], b.iloc[:450][cols], check_exact=False, rtol=1e-9)


def test_supervised_has_no_nan_and_int_labels(ohlcv):
    sup = make_supervised(build_features(ohlcv))
    assert not sup.isna().any().any()
    assert set(sup["target_direction"].unique()) <= {0, 1}
    assert len(sup) < len(ohlcv)
