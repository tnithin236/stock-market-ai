"""Performance metrics for a daily backtest."""
from typing import Dict

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_from_returns(net, rf_annual: float = 0.0) -> np.ndarray:
    """Annualised Sharpe along the last axis; 0 when volatility is zero."""
    net = np.asarray(net, dtype=float)
    excess = net - rf_annual / TRADING_DAYS
    std = net.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(std > 0, excess.mean(axis=-1) / std * np.sqrt(TRADING_DAYS), 0.0)
    return out


def max_drawdown(equity_with_start) -> float:
    eq = np.asarray(equity_with_start, dtype=float)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def _trade_returns(position: np.ndarray, net: np.ndarray) -> np.ndarray:
    """Return of each contiguous same-direction holding period (compounded daily net returns)."""
    out, i, n = [], 0, len(position)
    while i < n:
        if position[i] != 0:
            j = i
            while j + 1 < n and position[j + 1] * position[i] > 0:
                j += 1
            out.append(np.prod(1.0 + net[i:j + 1]) - 1.0)
            i = j + 1
        else:
            i += 1
    return np.asarray(out, dtype=float)


def performance_metrics(bt: pd.DataFrame, initial_capital: float, rf_annual: float = 0.0) -> Dict[str, float]:
    net, pos = bt["net_ret"].to_numpy(), bt["position"].to_numpy()
    equity = bt["equity"].to_numpy()
    n = len(net)
    total = equity[-1] / initial_capital - 1.0
    years = n / TRADING_DAYS
    cagr = (equity[-1] / initial_capital) ** (1.0 / years) - 1.0 if equity[-1] > 0 else -1.0
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    downside = net[net < 0]
    dd_std = float(np.sqrt(np.mean(downside ** 2))) if len(downside) else 0.0
    sortino = float((net.mean() - rf_annual / TRADING_DAYS) / dd_std * np.sqrt(TRADING_DAYS)) if dd_std > 0 else 0.0
    mdd = max_drawdown(np.concatenate([[initial_capital], equity]))
    in_mkt = pos != 0
    trades = _trade_returns(pos, net)
    return {
        "final_value": float(equity[-1]),
        "total_return_%": total * 100,
        "CAGR_%": cagr * 100,
        "volatility_%": vol * 100,
        "sharpe": float(sharpe_from_returns(net, rf_annual)),
        "sortino": sortino,
        "max_drawdown_%": mdd * 100,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
        "win_rate_days_%": float(np.mean(net[in_mkt] > 0) * 100) if in_mkt.any() else 0.0,
        "n_trades": int(len(trades)),
        "trade_win_rate_%": float(np.mean(trades > 0) * 100) if len(trades) else 0.0,
        "exposure_%": float(np.mean(np.abs(pos)) * 100),
        "turnover": float(bt["turnover"].sum()),
        "cost_drag_%": float(bt["cost_ret"].sum() * 100),
    }
