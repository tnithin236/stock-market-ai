"""Is a strategy better than luck? Compare it with many random strategies of the same average exposure."""
from typing import Tuple

import numpy as np

from src.backtesting.engine import MarketData, net_returns
from src.backtesting.metrics import sharpe_from_returns


def random_null(market: MarketData, exposure: float, n_trials: int, cost_bps: float, execution: str,
                allow_short: bool, seed: int, rf_annual: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sharpe, total_return) of `n_trials` random strategies."""
    rng = np.random.default_rng(seed)
    T = len(market.close)
    if allow_short:
        pos = rng.choice([-1.0, 1.0], size=(n_trials, T))
    else:
        pos = (rng.random((n_trials, T)) < min(max(exposure, 0.0), 1.0)).astype(float)
    net, *_ = net_returns(pos, market, cost_bps, execution)
    return sharpe_from_returns(net, rf_annual), np.prod(1.0 + net, axis=1) - 1.0


def null_summary(strategy_sharpe: float, null_sharpe: np.ndarray) -> dict:
    return {
        "null_median_sharpe": float(np.median(null_sharpe)),
        "beats_random_%": float(np.mean(null_sharpe < strategy_sharpe) * 100),
        "p_value": float(np.mean(null_sharpe >= strategy_sharpe)),   # one-sided
    }


def shift_null(market: MarketData, positions, n_trials: int, cost_bps: float, execution: str, seed: int,
               rf_annual: float = 0.0, min_shift: int = 21) -> np.ndarray:
    """Sharpe of the SAME position series circularly shifted by random amounts.

    Unlike i.i.d. random positions, this keeps the strategy's exposure, number of trades and holding periods
    (so trading costs are matched) and only breaks the alignment with the returns it is trying to time.
    """
    pos = np.asarray(positions, dtype=float)
    T = len(pos)
    if T <= 2 * min_shift + 1:
        raise ValueError("Not enough days for a shift test.")
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, T - min_shift, size=n_trials)
    matrix = np.stack([np.roll(pos, int(k)) for k in shifts])
    net, *_ = net_returns(matrix, market, cost_bps, execution)
    return sharpe_from_returns(net, rf_annual)
