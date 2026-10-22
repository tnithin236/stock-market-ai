"""Deterministic synthetic OHLCV data for tests, demos and offline runs (NOT real market data)."""
import numpy as np
import pandas as pd


def make_ohlcv(n: int = 600, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="Date")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n))), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
