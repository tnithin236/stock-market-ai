"""Turn model outputs into target positions (fraction of equity, decided at the close of day t)."""
import numpy as np
import pandas as pd


def _floor(allow_short: bool) -> float:
    return -1.0 if allow_short else 0.0


def positions_from_prob(prob, threshold: float = 0.5, sizing: str = "binary",
                        scale: float = 10.0, allow_short: bool = False) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    lo = _floor(allow_short)
    if sizing == "binary":
        return np.where(prob > threshold, 1.0, lo)
    if sizing == "scaled":
        return np.clip((prob - threshold) * scale, lo, 1.0)
    raise ValueError("sizing must be 'binary' or 'scaled'")


def positions_from_return(pred_return, allow_short: bool = False) -> np.ndarray:
    """Long when the model forecasts a positive next-day return."""
    return np.where(np.asarray(pred_return, dtype=float) > 0, 1.0, _floor(allow_short))


def buy_and_hold(n: int) -> np.ndarray:
    return np.ones(n)


def vol_filter_positions(vol: pd.Series, min_periods: int = 250) -> pd.Series:
    """Long when recent realised volatility is at or below its own historical median, flat otherwise.

    Fully causal: the median uses only data up to each day (expanding window). A direction-free baseline that
    simply sits out turbulent periods, used to check whether an ML strategy is merely doing volatility timing.
    """
    threshold = vol.expanding(min_periods=min_periods).median()
    return (~(vol > threshold)).astype(float)          # NaN comparisons are False -> long during warm-up
