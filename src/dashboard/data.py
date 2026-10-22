"""Data helpers for the Streamlit dashboard (no Streamlit import, so they are unit-testable)."""
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.loader import load_stock_data, normalize_ticker
from src.data.synthetic import make_ohlcv
from src.features import indicators as ind


def get_market_data(ticker: str, start, end, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Validated OHLCV for the chosen range. STOCKAI_SYNTHETIC=1 swaps in fake data (demos / offline tests)."""
    t = normalize_ticker(ticker)
    if os.environ.get("STOCKAI_SYNTHETIC") == "1":
        df = make_ohlcv(1500, seed=sum(map(ord, t)) % 1000)
        sliced = df.loc[str(start):str(end)]
        return sliced if len(sliced) else df
    d = cfg["data"]
    return load_stock_data(t, str(start), str(end) if end else None, d["interval"], d["raw_dir"],
                           d["cache_max_age_hours"], force_refresh=False)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Columns used by the indicator charts (all causal; see src/features/indicators.py)."""
    out = df.copy()
    close = out["Close"]
    out["sma_20"], out["sma_50"] = ind.sma(close, 20), ind.sma(close, 50)
    out = out.join(ind.bollinger(close, 20, 2.0)[["bb_mid", "bb_upper", "bb_lower"]])
    out["rsi_14"] = ind.rsi(close, 14)
    return out.join(ind.macd(close))


def summary_stats(df: pd.DataFrame) -> Dict[str, float]:
    close = df["Close"]
    rets = close.pct_change().dropna()
    return {
        "latest_close": float(close.iloc[-1]),
        "last_day_change_pct": float(rets.iloc[-1] * 100) if len(rets) else 0.0,
        "period_return_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "annualized_volatility_pct": float(rets.std() * np.sqrt(252) * 100) if len(rets) > 1 else 0.0,
        "max_drawdown_pct": float((close / close.cummax() - 1).min() * 100),
        "n_days": int(len(df)),
    }
