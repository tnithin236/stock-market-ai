"""Backtest figures (matplotlib, PNG)."""
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_equity(curves: Dict[str, pd.DataFrame], title: str, path) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, bt in curves.items():
        ax.plot(bt.index, bt["equity"], label=name, lw=2.2 if name == "buy_and_hold" else 1.3,
                color="black" if name == "buy_and_hold" else None)
    ax.set_title(title)
    ax.set_ylabel("Portfolio value")
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    return _save(fig, path)


def plot_drawdowns(curves: Dict[str, pd.DataFrame], title: str, path) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, bt in curves.items():
        ax.plot(bt.index, bt["drawdown"] * 100, label=name, lw=2.0 if name == "buy_and_hold" else 1.1,
                color="black" if name == "buy_and_hold" else None)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    return _save(fig, path)


def plot_signals(price: pd.Series, events: pd.DataFrame, title: str, path) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(price.index, price.values, color="#334155", lw=1.2, label="Close")
    buys, sells = events[events["action"] == "BUY"], events[events["action"] == "SELL"]
    ax.scatter(buys["signal_date"], buys["price"], marker="^", color="#16a34a", s=28, label="BUY", zorder=3)
    ax.scatter(sells["signal_date"], sells["price"], marker="v", color="#dc2626", s=28, label="SELL", zorder=3)
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, path)


def plot_cost_sensitivity(table: pd.DataFrame, title: str, path) -> str:
    """table: rows = strategies, columns = cost in bps, values = total return %."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [float(c) for c in table.columns]
    for name, row in table.iterrows():
        ax.plot(xs, row.values, marker="o", label=name, lw=2.2 if name == "buy_and_hold" else 1.2,
                color="black" if name == "buy_and_hold" else None)
    ax.set_xlabel("One-way cost (bps)")
    ax.set_ylabel("Total return (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_null_distribution(null_sharpe: np.ndarray, strategy_sharpe: float, name: str, path) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(null_sharpe, bins=40, color="#cbd5e1")
    ax.axvline(strategy_sharpe, color="#dc2626", lw=2, label=f"{name} ({strategy_sharpe:.2f})")
    ax.set_xlabel("Sharpe ratio of random strategies (same average exposure)")
    ax.set_ylabel("count")
    ax.set_title("Is the strategy better than luck?")
    ax.legend()
    return _save(fig, path)
