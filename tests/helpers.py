"""Synthetic market data so tests run offline and deterministically."""
import numpy as np
import pandas as pd


def make_ohlcv(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n, name="Date")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    close = pd.Series(close, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
