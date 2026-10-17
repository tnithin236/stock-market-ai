"""Vectorised daily backtest engine.

Timing (no look-ahead):
  * A signal for day t+1 is computed from features known at the CLOSE of day t.
  * execution="next_open": we trade at the OPEN of day t+1. The old position earns the overnight gap
    (close t -> open t+1); the new position earns the intraday move (open t+1 -> close t+1).
  * execution="same_close": idealised - we trade at the close of day t and earn close t -> close t+1.
Costs: proportional to turnover (|new position - old position|), charged on equity.
Position = fraction of equity invested (+1 long, 0 flat, -1 short).
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

EXECUTIONS = ("next_open", "same_close")


@dataclass
class MarketData:
    signal_dates: pd.DatetimeIndex   # day t (signal made at its close)
    pnl_dates: pd.DatetimeIndex      # day t+1 (when the P&L is earned)
    close: np.ndarray                # close of day t
    overnight: np.ndarray            # open(t+1) / close(t) - 1
    intraday: np.ndarray             # close(t+1) / open(t+1) - 1
    ctc: np.ndarray                  # close(t+1) / close(t) - 1


def build_market_data(full: pd.DataFrame, signal_dates) -> MarketData:
    """`full` must contain every trading day INCLUDING the day after the last signal day."""
    signal_dates = pd.DatetimeIndex(signal_dates)
    nxt_open = full["Open"].shift(-1).loc[signal_dates]
    nxt_close = full["Close"].shift(-1).loc[signal_dates]
    nxt_date = pd.Series(full.index, index=full.index).shift(-1).loc[signal_dates]
    close = full["Close"].loc[signal_dates]
    if nxt_open.isna().any() or nxt_close.isna().any() or nxt_date.isna().any():
        raise ValueError("Missing next-day prices for some signal dates.")
    return MarketData(
        signal_dates=signal_dates,
        pnl_dates=pd.DatetimeIndex(nxt_date),
        close=close.to_numpy(float),
        overnight=(nxt_open / close - 1.0).to_numpy(float),
        intraday=(nxt_close / nxt_open - 1.0).to_numpy(float),
        ctc=(nxt_close / close - 1.0).to_numpy(float),
    )


def net_returns(pos, market: MarketData, cost_bps: float, execution: str
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-day (net, gross, cost, turnover). `pos` may be 1-D or (n_sims, T); output is 2-D."""
    if execution not in EXECUTIONS:
        raise ValueError(f"execution must be one of {EXECUTIONS}")
    pos = np.atleast_2d(np.asarray(pos, dtype=float))
    if pos.shape[1] != len(market.close):
        raise ValueError("positions and market data have different lengths")
    prev = np.concatenate([np.zeros((pos.shape[0], 1)), pos[:, :-1]], axis=1)
    turnover = np.abs(pos - prev)
    cost = turnover * (cost_bps / 1e4)                      # fraction of equity paid at the trade
    if execution == "next_open":
        # old position earns the overnight gap, the trade (and its cost) happens at the open,
        # then the new position earns the intraday move. Exact compounding, no approximation.
        growth_gross = (1.0 + prev * market.overnight) * (1.0 + pos * market.intraday)
    else:
        growth_gross = 1.0 + pos * market.ctc
    return growth_gross * (1.0 - cost) - 1.0, growth_gross - 1.0, cost, turnover


def run_backtest(market: MarketData, positions, initial_capital: float = 100_000.0,
                 cost_bps: float = 5.0, execution: str = "next_open") -> pd.DataFrame:
    pos = np.asarray(positions, dtype=float)
    net, gross, cost, turn = (a[0] for a in net_returns(pos, market, cost_bps, execution))
    equity = initial_capital * np.cumprod(1.0 + net)
    start = np.concatenate([[initial_capital], equity])
    drawdown = (start / np.maximum.accumulate(start) - 1.0)[1:]
    return pd.DataFrame({
        "position": pos, "gross_ret": gross, "cost_ret": cost, "net_ret": net,
        "turnover": turn, "equity": equity, "drawdown": drawdown,
    }, index=market.pnl_dates)


def trade_events(market: MarketData, positions) -> pd.DataFrame:
    """BUY/SELL markers where the target position changes (plotted at the signal day's close)."""
    pos = np.asarray(positions, dtype=float)
    prev = np.concatenate([[0.0], pos[:-1]])
    delta = pos - prev
    idx = np.nonzero(delta)[0]
    return pd.DataFrame({
        "signal_date": market.signal_dates[idx],
        "action": np.where(delta[idx] > 0, "BUY", "SELL"),
        "price": market.close[idx],
        "new_position": pos[idx],
    })
