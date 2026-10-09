"""Project CLI. Step 1 provides `eda`; later steps add train / evaluate / backtest."""
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
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
