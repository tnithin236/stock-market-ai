"""Robustness study: the same walk-forward XGBoost experiment on many tickers, with multiple-testing correction.

Question: if one ticker looks good, is that skill or the luck you expect when you try many? With N tickers and
alpha = 0.05 you expect about 0.05 * N "significant" results from pure chance, so we also report Holm-Bonferroni
and Benjamini-Hochberg corrected p-values.

Caveats printed with the results: the tickers are all large US stocks (survivors) that move together, so N tickers
are not N independent experiments, and the p-values use the daily-independence assumption (slightly optimistic).
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.backtesting.engine import build_market_data, run_backtest
from src.backtesting.metrics import performance_metrics
from src.backtesting.signals import buy_and_hold, positions_from_prob
from src.config import resolve_path
from src.evaluation.reporting import fmt, json_default
from src.evaluation.stats import accuracy_vs_rate_pvalue, bootstrap_auc_ci, r2_vs_zero
from src.features.engineering import feature_columns
from src.models.xgb_models import merged_xgb_config
from src.training.dataset import load_supervised
from src.training.walk_forward import fold_sign_test, merged_wf_config, walk_forward_predictions
from src.visualization import model_plots as mp

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM", "JNJ", "WMT", "KO", "PG"]


def auc_pvalue(y_true, score) -> float:
    """One-sided p-value that AUC > 0.5 (Mann-Whitney U test on scores of up days vs down days)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    up, down = s[y == 1], s[y == 0]
    if len(up) == 0 or len(down) == 0:
        return float("nan")
    return float(stats.mannwhitneyu(up, down, alternative="greater").pvalue)


def holm_bonferroni(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the chance of ANY false positive)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (controls the expected share of false discoveries)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for rank_from_top, idx in enumerate(order):
        rank = m - rank_from_top
        running = min(running, p[idx] * m / rank)
        adj[idx] = min(1.0, running)
    return adj


def evaluate_ticker(cfg: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    wc, xc = merged_wf_config(cfg), merged_xgb_config(cfg)
    ticker, full, sup = load_supervised(cfg, ticker)
    cols = feature_columns(sup, scale_free_only=(wc["feature_set"] == "scale_free"))
    preds, folds = walk_forward_predictions(
        sup, cols, xc["params"], xc["n_estimators"], xc["early_stopping_rounds"], xc["seed"], xc["use_class_weight"],
        wc["initial_train_days"], wc["test_days"], wc["gap"], wc["val_frac"])
    frame = sup.loc[preds.index].join(preds)
    y = frame["target_direction"].to_numpy().astype(int)
    raw = frame["raw_prob"].to_numpy()
    n, k = len(y), int(((raw > 0.5) == y).sum())
    lo, hi = bootstrap_auc_ci(y, raw, n_boot=500)
    sign = fold_sign_test(folds["auc"])

    bcfg = cfg.get("backtest") or {}
    ic, cost, ex = float(bcfg.get("initial_capital", 100000)), float(bcfg.get("cost_bps", 5)), bcfg.get("execution", "next_open")
    market = build_market_data(full, frame.index)
    bt_clf = performance_metrics(run_backtest(market, positions_from_prob(raw), ic, cost, ex), ic)
    bt_bh = performance_metrics(run_backtest(market, buy_and_hold(n), ic, cost, ex), ic)
    y_ret = frame["target_return"].to_numpy()
    naive_rmse = float(np.sqrt(np.mean(y_ret ** 2)))
    model_rmse = float(np.sqrt(np.mean((frame["pred_return"].to_numpy() - y_ret) ** 2)))
    return {
        "ticker": ticker, "oos_days": n, "folds": int(len(folds)),
        "auc": float(roc_auc_score(y, raw)), "auc_lo": lo, "auc_hi": hi, "auc_p": auc_pvalue(y, raw),
        "accuracy": k / n, "always_up_acc": float(y.mean()),
        "acc_p_vs_up": accuracy_vs_rate_pvalue(k, n, max(float(y.mean()), 1 - float(y.mean()))),
        "folds_auc_gt_half": sign["folds_auc_above_0.5"], "fold_sign_p": sign["p_value"],
        "rmse_vs_naive": model_rmse / naive_rmse if naive_rmse else float("nan"),
        "oos_r2_ret": r2_vs_zero(y_ret, frame["pred_return"].to_numpy()),
        "clf_sharpe": bt_clf["sharpe"], "bh_sharpe": bt_bh["sharpe"],
        "clf_return_%": bt_clf["total_return_%"], "bh_return_%": bt_bh["total_return_%"],
        "clf_max_dd_%": bt_clf["max_drawdown_%"], "bh_max_dd_%": bt_bh["max_drawdown_%"],
    }


def run_multi_ticker(cfg: Dict[str, Any], tickers: Optional[List[str]] = None, make_plots: bool = True,
                     save: bool = True) -> Dict[str, Any]:
    mc = cfg.get("multi_ticker") or {}
    tickers = [t.upper() for t in (tickers or mc.get("tickers") or DEFAULT_TICKERS)]
    alpha = float(mc.get("alpha", 0.05))
    print(f"\n=== Multi-ticker walk-forward robustness study: {len(tickers)} tickers ===")
    rows, failed = [], {}
    for t in tickers:
        try:
            print(f"  evaluating {t} ...")
            rows.append(evaluate_ticker(cfg, t))
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the study
            failed[t] = f"{type(exc).__name__}: {str(exc)[:200]}"
            print(f"  skipped {t}: {failed[t]}")
    if len(rows) < 2:
        raise ValueError("Need at least two tickers to evaluate for a robustness study.")

    df = pd.DataFrame(rows).set_index("ticker")
    df["auc_p_holm"] = holm_bonferroni(df["auc_p"].to_numpy())
    df["auc_p_bh"] = benjamini_hochberg(df["auc_p"].to_numpy())
    n = len(df)
    nominal = int((df["auc_p"] < alpha).sum())
    holm_sig = int((df["auc_p_holm"] < alpha).sum())
    bh_sig = int((df["auc_p_bh"] < alpha).sum())
    above = int((df["auc"] > 0.5).sum())
    sign_p = float(stats.binomtest(above, n, 0.5, alternative="greater").pvalue)
    t_stat = stats.ttest_1samp(df["auc"], 0.5, alternative="greater")

    print("\n=== Per-ticker walk-forward direction results (pooled out-of-sample) ===")
    print(fmt(df[["oos_days", "auc", "auc_lo", "auc_hi", "auc_p", "auc_p_holm", "accuracy", "always_up_acc",
                  "folds_auc_gt_half", "fold_sign_p"]], 4))
    print("\n=== Per-ticker long/flat strategy vs buy-and-hold (5 bps, next-open) ===")
    show = df[["clf_sharpe", "bh_sharpe", "clf_return_%", "bh_return_%", "clf_max_dd_%", "bh_max_dd_%"]]
    print(fmt(show, 2))
    print("\n=== Regression (next-day return) ===")
    print(fmt(df[["rmse_vs_naive", "oos_r2_ret"]], 4))

    expected = alpha * n
    print("\n=== Summary ===")
    print(f"Tickers evaluated: {n}" + (f" | skipped: {sorted(failed)}" if failed else ""))
    print(f"Mean AUC {df['auc'].mean():.3f} (min {df['auc'].min():.3f}, max {df['auc'].max():.3f}); "
          f"AUC > 0.5 for {above} of {n} tickers (sign-test p = {sign_p:.3f}); mean-AUC t-test p = {t_stat.pvalue:.3f}")
    print(f"Nominally significant at alpha = {alpha}: {nominal} of {n} (about {expected:.1f} expected by chance alone)")
    print(f"After Holm-Bonferroni correction: {holm_sig} | after Benjamini-Hochberg FDR: {bh_sig}")
    better = int((df["clf_sharpe"] > df["bh_sharpe"]).sum())
    print(f"The ML long/flat strategy had a higher Sharpe than buy-and-hold on {better} of {n} tickers.")
    print("\nCaveats: all tickers are large US companies that survived to today (survivorship bias) and they move "
          "together, so they are not independent experiments. Daily-independence assumptions make p-values slightly "
          "optimistic. Experimental research, not financial advice.")

    results = {"tickers": list(df.index), "failed": failed, "alpha": alpha,
               "summary": {"mean_auc": float(df["auc"].mean()), "n_auc_gt_half": above, "sign_test_p": sign_p,
                           "mean_auc_ttest_p": float(t_stat.pvalue), "nominal_significant": nominal,
                           "expected_by_chance": expected, "holm_significant": holm_sig, "bh_significant": bh_sig,
                           "strategy_sharpe_beats_bh": better},
               "per_ticker": df.reset_index().to_dict("records")}
    figures = []
    if make_plots:
        fig_dir = Path(resolve_path(cfg["paths"]["figures_dir"]))
        figures.append(mp.plot_multi_ticker_auc(df, alpha, fig_dir / "multi_ticker_auc.png"))
        print(f"\nSaved figure: {figures[0]}")
    if save:
        reports = Path(resolve_path(cfg["paths"].get("reports_dir", "reports")))
        (reports / "multi_ticker_results.json").write_text(json.dumps(results, indent=2, default=json_default))
        df.to_csv(reports / "multi_ticker_results.csv")
        print(f"Saved results to {reports}")
    results["figures"] = figures
    return results
