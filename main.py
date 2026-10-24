"""Project CLI. Commands: eda, features, baseline, xgboost, lstm, transformer, compare, backtest, walkforward."""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.data.loader import load_stock_data, normalize_ticker
from src.data.statistics import (add_returns, adf_report, class_balance,
                                 summary_statistics)
from src.visualization import eda_plots as viz

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def run_eda(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    d, e = cfg["data"], cfg["eda"]
    ticker = normalize_ticker(args.ticker or d["ticker"])
    start = args.start or d["start_date"]
    end = args.end or d["end_date"]

    df = load_stock_data(ticker, start, end, d["interval"], d["raw_dir"],
                         d["cache_max_age_hours"], force_refresh=args.refresh)
    df = add_returns(df)

    # ---- Text report ----
    print(f"\n=== {ticker}: first / last rows ===")
    print(pd.concat([df.head(3), df.tail(3)]).round(4))

    print("\n=== Summary statistics ===")
    for k, v in summary_statistics(df).items():
        print(f"{k:>26}: {v:.3f}" if isinstance(v, float) else f"{k:>26}: {v}")

    print("\n=== Stationarity (ADF test) ===")
    print(pd.DataFrame([adf_report(df["Close"], "Close price"),
                        adf_report(df["log_return"], "Log return")]).to_string(index=False))

    print("\n=== Class balance (next-day direction) ===")
    for k, v in class_balance(df).items():
        print(f"{k:>10}: {v:.2f}" if isinstance(v, float) else f"{k:>10}: {v}")

    # ---- Figures ----
    out = resolve_path(cfg["paths"]["figures_dir"])
    figs = {
        f"{ticker}_candlestick.html": viz.plot_candlestick(df, ticker, e["ma_windows"]),
        f"{ticker}_returns_dist.html": viz.plot_returns_distribution(df["return"], ticker),
        f"{ticker}_rolling_vol.html": viz.plot_rolling_volatility(df["return"], ticker, e["vol_window"]),
        f"{ticker}_acf.html": viz.plot_acf_comparison(df["return"], ticker, e["acf_lags"]),
        f"{ticker}_price_vs_returns.html": viz.plot_price_vs_returns(df, ticker),
    }
    for name, fig in figs.items():
        fig.write_html(Path(out) / name)
    print(f"\nSaved {len(figs)} interactive figures to {out}")


def run_features(args: argparse.Namespace) -> None:
    from src.features.engineering import feature_columns
    from src.training.dataset import load_supervised

    cfg = load_config(args.config)
    ticker, full, sup = load_supervised(cfg, args.ticker, args.start, args.end, args.refresh)
    cols = feature_columns(sup)
    out = resolve_path(cfg["data"]["processed_dir"]) / f"{ticker}_features.csv"
    sup.to_csv(out)
    print(f"\n=== {ticker}: feature table ===")
    print(f"full frame: {full.shape} | supervised: {sup.shape} | inputs: {len(cols)}")
    print("NaN cells left in supervised table:", int(sup.isna().sum().sum()))
    print("\nFeatures:\n  " + ", ".join(cols))
    print("\nLast 3 rows (selected columns):")
    show = [c for c in ["Close", "return", "rsi_14", "macd", "bb_pctb", "atr_14_pct",
                        "vol_20", "target_close", "target_direction"] if c in sup.columns]
    print(sup[show].tail(3).round(4).to_string())
    print(f"\nSaved to {out}")


def run_baseline(args: argparse.Namespace) -> None:
    from src.training.run_baselines import run_baselines

    cfg = load_config(args.config)
    run_baselines(cfg, args.ticker, args.start, args.end, args.refresh)


def run_xgboost_cmd(args: argparse.Namespace) -> None:
    from src.training.train_xgboost import run_xgboost

    cfg = load_config(args.config)
    run_xgboost(cfg, args.ticker, args.start, args.end, args.refresh,
                tune=args.tune, n_trials=args.trials, make_plots=not args.no_plots)


def run_lstm_cmd(args: argparse.Namespace) -> None:
    from src.training.train_lstm import run_lstm

    cfg = load_config(args.config)
    run_lstm(cfg, args.ticker, args.start, args.end, args.refresh,
             tune=args.tune, n_trials=args.trials, make_plots=not args.no_plots)


def run_transformer_cmd(args: argparse.Namespace) -> None:
    from src.training.train_transformer import run_transformer

    cfg = load_config(args.config)
    run_transformer(cfg, args.ticker, args.start, args.end, args.refresh,
                    tune=args.tune, n_trials=args.trials, make_plots=not args.no_plots)


def run_compare(args: argparse.Namespace) -> None:
    from src.evaluation.compare import build_comparison, read_test_periods, significant_models
    from src.visualization.model_plots import plot_model_comparison

    cfg = load_config(args.config)
    ticker = normalize_ticker(args.ticker or cfg["data"]["ticker"])
    reports = resolve_path(cfg["paths"].get("reports_dir", "reports"))
    reg, cls, found = build_comparison(reports, ticker)
    if not found:
        print("No result files found. Run `python main.py xgboost`, `lstm` and/or `transformer` first.")
        return
    periods = read_test_periods(reports, ticker)
    distinct = {p for p in periods.values() if p}
    if len(distinct) > 1 or any(p is None for p in periods.values()):
        print("\nWARNING: result files were NOT all produced on the same test period "
              f"({periods}). The table below mixes data snapshots, so re-run "
              "xgboost, lstm and transformer on the same data (see end_date in configs/config.yaml).")
    print(f"\n=== {ticker}: model comparison (TEST period) from {', '.join(found)} ===")
    print("\nRegression (next-day close; RMSE_vs_naive < 1 and oosR2_ret > 0 would mean an edge):")
    print(reg.round(4).to_string())
    print("\nClassification (next-day direction):")
    show = cls.copy()
    show["acc_CI95"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(show["acc_lo"], show["acc_hi"])]
    show["auc_CI95"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(show["auc_lo"], show["auc_hi"])]
    print(show[["accuracy", "acc_CI95", "p_vs_majority", "roc_auc", "auc_CI95", "brier"]].round(4).to_string())
    winners = significant_models(cls)
    print("\nModels with a statistically significant edge (p<0.05 vs majority AND AUC interval above 0.5):",
          winners if winners else "none")
    out_csv = Path(reports) / f"model_comparison_{ticker}.csv"
    cls.to_csv(out_csv)
    fig = plot_model_comparison(cls, Path(resolve_path(cfg["paths"]["figures_dir"])) / f"{ticker}_model_comparison.png",
                                f"{ticker}: test-set model comparison (95% intervals)")
    print(f"\nSaved {out_csv}\nSaved {fig}")


def run_backtest_cmd(args: argparse.Namespace) -> None:
    from src.backtesting.runner import run_backtest_study

    cfg = load_config(args.config)
    run_backtest_study(cfg, args.ticker, args.cost_bps, args.execution, args.sizing,
                       True if args.allow_short else None, args.plot_strategy, make_plots=not args.no_plots)


def run_walkforward_cmd(args: argparse.Namespace) -> None:
    from src.training.walk_forward import run_walk_forward

    cfg = load_config(args.config)
    run_walk_forward(cfg, args.ticker, args.initial_days, args.test_days, make_plots=not args.no_plots)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stock Market AI project CLI")
    p.add_argument("--config", default="configs/config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    eda = sub.add_parser("eda", help="Download data and run exploratory analysis")
    eda.add_argument("--ticker", default=None)
    eda.add_argument("--start", default=None)
    eda.add_argument("--end", default=None)
    eda.add_argument("--refresh", action="store_true", help="Ignore the local cache")
    eda.set_defaults(func=run_eda)

    for name, fn, helptext in (
        ("features", run_features, "Build the leakage-safe feature table"),
        ("baseline", run_baseline, "Chronological split + naive baselines"),
    ):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--ticker", default=None)
        sp.add_argument("--start", default=None)
        sp.add_argument("--end", default=None)
        sp.add_argument("--refresh", action="store_true", help="Ignore the local cache")
        sp.set_defaults(func=fn)

    xg = sub.add_parser("xgboost", help="Train/evaluate XGBoost (regression + classification)")
    xg.add_argument("--ticker", default=None)
    xg.add_argument("--start", default=None)
    xg.add_argument("--end", default=None)
    xg.add_argument("--refresh", action="store_true", help="Ignore the local cache")
    xg.add_argument("--tune", action="store_true", help="Run Optuna tuning (validation only)")
    xg.add_argument("--trials", type=int, default=None, help="Optuna trials (default from config)")
    xg.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    xg.set_defaults(func=run_xgboost_cmd)

    ls = sub.add_parser("lstm", help="Train/evaluate an LSTM (regression + classification)")
    ls.add_argument("--ticker", default=None)
    ls.add_argument("--start", default=None)
    ls.add_argument("--end", default=None)
    ls.add_argument("--refresh", action="store_true", help="Ignore the local cache")
    ls.add_argument("--tune", action="store_true", help="Run Optuna tuning (validation only)")
    ls.add_argument("--trials", type=int, default=None, help="Optuna trials (default from config)")
    ls.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    ls.set_defaults(func=run_lstm_cmd)

    tf = sub.add_parser("transformer", help="Train/evaluate a Transformer (regression + classification)")
    tf.add_argument("--ticker", default=None)
    tf.add_argument("--start", default=None)
    tf.add_argument("--end", default=None)
    tf.add_argument("--refresh", action="store_true", help="Ignore the local cache")
    tf.add_argument("--tune", action="store_true", help="Run Optuna tuning (validation only)")
    tf.add_argument("--trials", type=int, default=None, help="Optuna trials (default from config)")
    tf.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    tf.set_defaults(func=run_transformer_cmd)

    cp = sub.add_parser("compare", help="Compare all trained models on the test period")
    cp.add_argument("--ticker", default=None)
    cp.set_defaults(func=run_compare)

    bt = sub.add_parser("backtest", help="Backtest model signals vs buy-and-hold on the test period")
    bt.add_argument("--ticker", default=None)
    bt.add_argument("--cost-bps", type=float, default=None, help="One-way cost in basis points")
    bt.add_argument("--execution", choices=["next_open", "same_close"], default=None)
    bt.add_argument("--sizing", choices=["binary", "scaled"], default=None)
    bt.add_argument("--allow-short", action="store_true", help="Allow short positions (no borrow cost modelled)")
    bt.add_argument("--plot-strategy", default="xgboost_clf", help="Strategy shown in the BUY/SELL signal plot")
    bt.add_argument("--no-plots", action="store_true")
    bt.set_defaults(func=run_backtest_cmd)

    wf = sub.add_parser("walkforward", help="Walk-forward (expanding window) validation of XGBoost")
    wf.add_argument("--ticker", default=None)
    wf.add_argument("--initial-days", type=int, default=None, help="Rows in the first training window")
    wf.add_argument("--test-days", type=int, default=None, help="Rows per out-of-sample block")
    wf.add_argument("--no-plots", action="store_true")
    wf.set_defaults(func=run_walkforward_cmd)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
