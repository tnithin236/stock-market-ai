"""Statistical EDA helpers: returns, summary stats, stationarity, class balance."""
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

TRADING_DAYS = 252


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple and log daily returns (uses only past/current prices, no leakage)."""
    out = df.copy()
    out["return"] = out["Close"].pct_change()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    return out


def max_drawdown(prices: pd.Series) -> float:
    running_max = prices.cummax()
    return float((prices / running_max - 1).min())


def summary_statistics(df: pd.DataFrame) -> Dict[str, float]:
    r = df["return"].dropna()
    return {
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "n_days": int(len(df)),
        "mean_daily_return_%": r.mean() * 100,
        "daily_volatility_%": r.std() * 100,
        "annualized_return_%": r.mean() * TRADING_DAYS * 100,
        "annualized_volatility_%": r.std() * np.sqrt(TRADING_DAYS) * 100,
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),  # 0 for a normal distribution
        "best_day_%": r.max() * 100,
        "worst_day_%": r.min() * 100,
        "max_drawdown_%": max_drawdown(df["Close"]) * 100,
    }


def adf_report(series: pd.Series, name: str) -> Dict[str, object]:
    """Augmented Dickey-Fuller test. H0: the series has a unit root (non-stationary)."""
    stat, pvalue, *_ = adfuller(series.dropna(), autolag="AIC", result_object=False)
    return {
        "series": name,
        "adf_statistic": round(float(stat), 4),
        "p_value": round(float(pvalue), 4),
        "stationary_at_5%": bool(pvalue < 0.05),
    }


def class_balance(df: pd.DataFrame) -> Dict[str, float]:
    """Share of days where the NEXT day's close is above today's close.

    The final row has no 'next day', so it is dropped instead of being
    silently labeled as 'down'.
    """
    nxt = df["Close"].shift(-1)
    valid = nxt.notna()
    up = (nxt[valid] > df["Close"][valid]).astype(int)
    return {
        "n_samples": int(len(up)),
        "up_%": float(up.mean() * 100),
        "down_%": float((1 - up.mean()) * 100),
    }
