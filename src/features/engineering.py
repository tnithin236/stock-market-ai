"""Leakage-safe feature engineering.

Rules this module follows:
1. Every feature at row t uses only data from rows <= t (see indicators.py).
2. Targets look forward (t+1) and live in columns prefixed with `target_`.
   They are NEVER included in feature_columns().
3. Nothing here is fitted on data (no scaling, no imputation statistics).
   Fitting happens later, on the training split only.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.features import indicators as ind

logger = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
TARGET_PREFIX = "target_"

DEFAULT_FEATURE_CONFIG: Dict[str, Any] = {
    "sma_windows": [10, 20, 50],
    "ema_windows": [10, 20, 50],
    "rsi_window": 14,
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"window": 20, "num_std": 2.0},
    "atr_window": 14,
    "vol_windows": [10, 20],
    "momentum_windows": [5, 10],
    "price_lags": [1, 2, 3, 5],
    "return_lags": [1, 2, 3, 5],
    "volume_ma_window": 20,
}

_SCALE_DEPENDENT_PREFIXES = ("sma_", "ema_", "close_lag_", "bb_mid", "bb_upper",
                             "bb_lower", "macd", "atr_")
_SCALE_FREE_SUFFIXES = ("_ratio", "_pct", "_pctb", "_width")


def _merge_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_FEATURE_CONFIG.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def build_features(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Return OHLCV + engineered features + next-day targets.

    The final row has NaN targets (tomorrow is unknown). Keep it for live
    prediction; make_supervised() removes it for training/evaluation.
    """
    p = _merge_config(cfg)
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted ascending (chronological).")

    out = df[OHLCV].copy()
    close, high, low, volume = out["Close"], out["High"], out["Low"], out["Volume"]

    # Returns
    out["return"] = close.pct_change()
    out["log_return"] = np.log(close / close.shift(1))

    # Trend: moving averages and distance of price from them
    for w in p["sma_windows"]:
        out[f"sma_{w}"] = ind.sma(close, w)
        out[f"close_sma_{w}_ratio"] = close / out[f"sma_{w}"] - 1.0
    for w in p["ema_windows"]:
        out[f"ema_{w}"] = ind.ema(close, w)
        out[f"close_ema_{w}_ratio"] = close / out[f"ema_{w}"] - 1.0

    # Momentum oscillators
    out[f"rsi_{p['rsi_window']}"] = ind.rsi(close, p["rsi_window"])
    out = out.join(ind.macd(close, **p["macd"]))
    for w in p["momentum_windows"]:
        out[f"mom_{w}"] = close / close.shift(w) - 1.0

    # Volatility
    out = out.join(ind.bollinger(close, **p["bollinger"]))
    aw = p["atr_window"]
    out[f"atr_{aw}"] = ind.atr(high, low, close, aw)
    out[f"atr_{aw}_pct"] = out[f"atr_{aw}"] / close
    for w in p["vol_windows"]:
        out[f"vol_{w}"] = out["log_return"].rolling(w, min_periods=w).std()

    # Volume
    out["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    vw = p["volume_ma_window"]
    vol_ma = volume.rolling(vw, min_periods=vw).mean()
    out[f"volume_ratio_{vw}"] = (volume / vol_ma.replace(0.0, np.nan))

    # Lags
    for k in p["price_lags"]:
        out[f"close_lag_{k}"] = close.shift(k)
    for k in p["return_lags"]:
        out[f"ret_lag_{k}"] = out["return"].shift(k)

    # Targets (look FORWARD by one trading day; never used as features)
    nxt = close.shift(-1)
    out[f"{TARGET_PREFIX}close"] = nxt
    out[f"{TARGET_PREFIX}return"] = nxt / close - 1.0
    out[f"{TARGET_PREFIX}direction"] = (nxt > close).astype(float).where(nxt.notna())

    return out


def is_scale_dependent(col: str) -> bool:
    """True for columns measured in price units (or raw volume) that trend over time."""
    if col in ("Open", "High", "Low", "Close", "Volume"):
        return True
    if col.endswith(_SCALE_FREE_SUFFIXES):
        return False
    return col.startswith(_SCALE_DEPENDENT_PREFIXES)


def feature_columns(df: pd.DataFrame, scale_free_only: bool = False) -> List[str]:
    """All model inputs (everything except targets), optionally only scale-free ones."""
    cols = [c for c in df.columns if not c.startswith(TARGET_PREFIX)]
    if scale_free_only:
        cols = [c for c in cols if not is_scale_dependent(c)]
    return cols


def make_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Drop warm-up rows (indicator NaNs) and the last row (unknown target)."""
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    out = cleaned.dropna()
    out = out.assign(**{f"{TARGET_PREFIX}direction": out[f"{TARGET_PREFIX}direction"].astype(int)})
    logger.info("Supervised dataset: %d rows kept, %d dropped (warm-up / unknown target)",
                len(out), len(df) - len(out))
    return out
