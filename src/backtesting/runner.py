"""Step 6: backtest ML signals against buy-and-hold on the TEST period.

EXPERIMENTAL RESEARCH / EDUCATIONAL CODE. NOT FINANCIAL ADVICE. Past or simulated performance
does not predict future results; the models in this project show no statistically significant edge.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.backtesting.engine import build_market_data, run_backtest, trade_events
from src.backtesting.metrics import performance_metrics
from src.backtesting.random_baseline import null_summary, random_null
from src.backtesting.signals import buy_and_hold, positions_from_prob, positions_from_return
from src.config import resolve_path
from src.evaluation.predictions import load_predictions
from src.training.dataset import load_supervised, split_dataset
from src.visualization import backtest_plots as bp

logger = logging.getLogger(__name__)

DEFAULT_BACKTEST: Dict[str, Any] = {
    "initial_capital": 100_000, "cost_bps": 5.0, "execution": "next_open", "sizing": "binary",
    "scale": 10.0, "threshold": 0.5, "allow_short": False, "cost_sensitivity_bps": [0, 5, 10, 25],
    "random_trials": 1000, "risk_free_annual": 0.0, "seed": 42,
}
MODEL_TAGS = ("xgboost", "lstm", "transformer")
DISCLAIMER = ("EXPERIMENTAL RESEARCH / EDUCATIONAL USE ONLY - NOT FINANCIAL ADVICE. Simulated results ignore taxes, "
              "market impact, borrow costs and many real-world frictions, and past performance does not predict "
              "future results.")


def merged_backtest_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULT_BACKTEST)
    out.update(cfg.get("backtest") or {})
    return out


def build_strategies(test_sup: pd.DataFrame, preds: Dict[str, pd.DataFrame], bc: Dict[str, Any]) -> Dict[str, np.ndarray]:
    n = len(test_sup)
    strategies: Dict[str, np.ndarray] = {
        "buy_and_hold": buy_and_hold(n),
        "momentum_rule": positions_from_return(test_sup["return"].to_numpy(), bc["allow_short"]),
    }
    for tag, p in preds.items():
        strategies[f"{tag}_clf"] = positions_from_prob(
            p["raw_prob"].to_numpy(), bc["threshold"], bc["sizing"], bc["scale"], bc["allow_short"])
        strategies[f"{tag}_reg"] = positions_from_return(p["pred_return"].to_numpy(), bc["allow_short"])
    return strategies


def run_backtest_study(
    cfg: Dict[str, Any],
    ticker: Optional[str] = None,
    cost_bps: Optional[float] = None,
    execution: Optional[str] = None,
    sizing: Optional[str] = None,
    allow_short: Optional[bool] = None,
    plot_strategy: str = "xgboost_clf",
    make_plots: bool = True,
    save: bool = True,
) -> Dict[str, Any]:
    bc = merged_backtest_config(cfg)
    if cost_bps is not None:
        bc["cost_bps"] = cost_bps
    if execution:
        bc["execution"] = execution
    if sizing:
        bc["sizing"] = sizing
    if allow_short is not None:
        bc["allow_short"] = allow_short

    ticker, full, sup = load_supervised(cfg, ticker)
    split = split_dataset(sup, cfg)
    test_sup = split.test
    market = build_market_data(full, test_sup.index)

    reports = Path(resolve_path(cfg["paths"].get("reports_dir", "reports")))
    preds: Dict[str, pd.DataFrame] = {}
    for tag in MODEL_TAGS:
        path = reports / f"predictions_{tag}_{ticker}.csv"
        if not path.exists():
            continue
        p = load_predictions(path, "test")
        if not p.index.equals(test_sup.index):
            raise ValueError(f"{path.name} covers different test dates than the current data snapshot. "
                             f"Re-run `python main.py {tag}` first.")
        preds[tag] = p
    if not preds:
        print("No prediction files found. Run `python main.py xgboost`, `lstm` and/or `transformer` first "
              "(they now save predictions for the backtest).")

    strategies = build_strategies(test_sup, preds, bc)
    ic, rf = float(bc["initial_capital"]), float(bc["risk_free_annual"])

    print(f"\n=== {ticker}: backtest on the TEST period {test_sup.index.min().date()} -> {market.pnl_dates.max().date()} ===")
    print(DISCLAIMER)
    print(f"Initial capital ${ic:,.0f} | one-way cost {bc['cost_bps']} bps | execution={bc['execution']} | "
          f"sizing={bc['sizing']} | shorting={'on' if bc['allow_short'] else 'off'}")

    curves = {name: run_backtest(market, pos, ic, bc["cost_bps"], bc["execution"]) for name, pos in strategies.items()}
    metrics = pd.DataFrame({name: performance_metrics(bt, ic, rf) for name, bt in curves.items()}).T
    print("\n=== Performance (after costs) ===")
    show = metrics.copy()
    show["final_value"] = show["final_value"].map(lambda v: f"${v:,.0f}")
    print(show.drop(columns=["sortino", "turnover"]).round(2).to_string())

    # ---- cost sensitivity ----
    levels = [float(x) for x in bc["cost_sensitivity_bps"]]
    ret_tbl, sharpe_tbl = {}, {}
    for name, pos in strategies.items():
        r, s = {}, {}
        for c in levels:
            m = performance_metrics(run_backtest(market, pos, ic, c, bc["execution"]), ic, rf)
            r[c], s[c] = m["total_return_%"], m["sharpe"]
        ret_tbl[name], sharpe_tbl[name] = r, s
    ret_df, sharpe_df = pd.DataFrame(ret_tbl).T, pd.DataFrame(sharpe_tbl).T
    print("\n=== Total return % by one-way cost (bps) ===")
    print(ret_df.round(2).to_string())
    print("\n=== Sharpe by one-way cost (bps) ===")
    print(sharpe_df.round(2).to_string())

    # ---- is it better than luck? ----
    null_rows, null_store = {}, {}
    for name, pos in strategies.items():
        if name == "buy_and_hold":
            continue
        exposure = float(np.mean(np.abs(pos)))
        sh, _tot = random_null(market, exposure, int(bc["random_trials"]), bc["cost_bps"], bc["execution"],
                               bool(bc["allow_short"]), int(bc["seed"]), rf)
        null_rows[name] = {"sharpe": metrics.loc[name, "sharpe"], **null_summary(metrics.loc[name, "sharpe"], sh)}
        null_store[name] = sh
    null_df = pd.DataFrame(null_rows).T
    n_tested = max(len(null_df), 1)
    null_df["p_bonferroni"] = (null_df["p_value"] * n_tested).clip(upper=1.0)   # we tried n_tested strategies
    print(f"\n=== Better than luck? ({int(bc['random_trials'])} random strategies with the same average exposure) ===")
    print(null_df.round(3).to_string())
    print("beats_random_% near 50 means indistinguishable from luck. With this many strategies compared, "
          "even a p-value below 0.05 can be a fluke (multiple comparisons).")

    bh = metrics.loc["buy_and_hold"]
    better = [n for n in null_df.index if metrics.loc[n, "sharpe"] > bh["sharpe"] and null_df.loc[n, "p_bonferroni"] < 0.05]
    print(f"\nStrategies with a higher Sharpe than buy-and-hold AND a Bonferroni-corrected p<0.05 vs random "
          f"(corrected for {n_tested} strategies tried): {better if better else 'none'}")
    print("Few trades means unreliable statistics: a Sharpe built on 2-5 trades is mostly luck.")

    figures = []
    if make_plots:
        fig_dir = Path(resolve_path(cfg["paths"]["figures_dir"]))
        figures.append(bp.plot_equity(curves, f"{ticker}: portfolio value after {bc['cost_bps']} bps costs (test period)",
                                      fig_dir / f"{ticker}_bt_equity.png"))
        figures.append(bp.plot_drawdowns(curves, f"{ticker}: drawdowns (test period)", fig_dir / f"{ticker}_bt_drawdown.png"))
        figures.append(bp.plot_cost_sensitivity(ret_df, f"{ticker}: total return vs trading cost",
                                                fig_dir / f"{ticker}_bt_cost_sensitivity.png"))
        if plot_strategy in strategies:
            ev = trade_events(market, strategies[plot_strategy])
            figures.append(bp.plot_signals(pd.Series(market.close, index=market.signal_dates), ev,
                                           f"{ticker}: {plot_strategy} BUY/SELL signals",
                                           fig_dir / f"{ticker}_bt_signals_{plot_strategy}.png"))
        if plot_strategy in null_store:
            figures.append(bp.plot_null_distribution(null_store[plot_strategy], float(metrics.loc[plot_strategy, "sharpe"]),
                                                     plot_strategy, fig_dir / f"{ticker}_bt_null_{plot_strategy}.png"))
        print("\nSaved figures:\n  " + "\n  ".join(figures))

    if save:
        metrics.to_csv(reports / f"backtest_summary_{ticker}.csv")
        pd.DataFrame({n: bt["equity"] for n, bt in curves.items()}).to_csv(reports / f"backtest_equity_{ticker}.csv")
        ret_df.to_csv(reports / f"backtest_cost_sensitivity_{ticker}.csv")
        null_df.to_csv(reports / f"backtest_null_{ticker}.csv")
        print(f"\nSaved backtest tables to {reports}")
    return {"metrics": metrics, "cost_returns": ret_df, "null": null_df, "figures": figures, "curves": curves}
