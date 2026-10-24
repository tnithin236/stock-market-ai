"""Walk-forward (expanding-window) validation of the XGBoost models.

For each block: train on everything before it (the last 15% of that window is held out, in time order, for early
stopping), predict the NEXT ~6 months, then roll forward and retrain. Stitching the blocks together gives ~15
out-of-sample periods instead of one, so results are less dependent on a single regime.
Defaults only (no tuning), so there is nothing to over-fit to the blocks.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.backtesting.engine import build_market_data, run_backtest
from src.backtesting.metrics import performance_metrics
from src.backtesting.random_baseline import null_summary, random_null
from src.backtesting.signals import buy_and_hold, positions_from_prob, positions_from_return
from src.config import resolve_path
from src.evaluation.reporting import classification_row, fmt, json_default, regression_row, tables
from src.features.engineering import feature_columns
from src.models.xgb_models import fit_model, make_model, merged_xgb_config, predict_proba_up, predict_return
from src.training.dataset import load_supervised
from src.visualization import backtest_plots as bp
from src.visualization import model_plots as mp

logger = logging.getLogger(__name__)

DEFAULT_WF = {"initial_train_days": 1008, "test_days": 126, "gap": 1, "val_frac": 0.15, "feature_set": "scale_free"}


def merged_wf_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULT_WF)
    out.update(cfg.get("walk_forward") or {})
    return out


def make_folds(n_rows: int, initial_train: int, test_size: int, gap: int = 1) -> List[Dict[str, int]]:
    """Expanding-window folds as row positions. Training rows are [0, train_end); testing rows are
    [test_start, test_end) with `gap` rows between them (the label of the last training row is the next day's price)."""
    if initial_train < 50 or test_size < 20 or gap < 0:
        raise ValueError("initial_train must be >= 50, test_size >= 20 and gap >= 0")
    folds, train_end = [], initial_train
    while True:
        start = train_end + gap
        if start >= n_rows:
            break
        end = min(start + test_size, n_rows)
        if end - start < max(20, test_size // 4):          # skip a tiny trailing block
            break
        folds.append({"train_end": train_end, "test_start": start, "test_end": end})
        train_end = end
    return folds


def walk_forward_predictions(
    sup: pd.DataFrame, cols: List[str], params: Dict[str, Any], n_estimators: int, early_stopping_rounds: int,
    seed: int, use_class_weight: bool, initial_train: int, test_size: int, gap: int = 1, val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (out-of-sample predictions indexed by date, per-fold summary)."""
    frames, fold_rows = [], []
    for k, f in enumerate(make_folds(len(sup), initial_train, test_size, gap)):
        train, test = sup.iloc[:f["train_end"]], sup.iloc[f["test_start"]:f["test_end"]]
        n_val = max(int(len(train) * val_frac), 50)
        fit, val = train.iloc[:len(train) - n_val - gap], train.iloc[len(train) - n_val:]
        n_up = int(fit["target_direction"].sum())
        spw = ((len(fit) - n_up) / n_up) if use_class_weight and n_up else None

        clf = make_model("classification", params, n_estimators, early_stopping_rounds, seed, spw)
        fit_model(clf, fit[cols], fit["target_direction"], val[cols], val["target_direction"])
        reg = make_model("regression", params, n_estimators, early_stopping_rounds, seed)
        fit_model(reg, fit[cols], fit["target_return"], val[cols], val["target_return"])

        raw, pred_ret = predict_proba_up(clf, test[cols]), predict_return(reg, test[cols])
        y = test["target_direction"].to_numpy().astype(int)
        frames.append(pd.DataFrame({"fold": k, "raw_prob": raw, "pred_return": pred_ret}, index=test.index))
        fold_rows.append({
            "fold": k, "train_rows": len(train), "test_start": str(test.index.min().date()),
            "test_end": str(test.index.max().date()), "n_test": len(test),
            "auc": float(roc_auc_score(y, raw)) if len(np.unique(y)) == 2 else float("nan"),
            "accuracy": float(np.mean((raw > 0.5) == y)), "always_up_accuracy": float(y.mean()),
            "clf_best_iter": int(clf.best_iteration),
        })
    if not frames:
        raise ValueError("No walk-forward folds: not enough data for the chosen initial_train_days / test_days.")
    return pd.concat(frames), pd.DataFrame(fold_rows).set_index("fold")


def fold_sign_test(fold_auc: pd.Series) -> Dict[str, float]:
    """If there were a real edge, AUC should be above 0.5 in most blocks. One-sided sign test."""
    a = fold_auc.dropna()
    k = int((a > 0.5).sum())
    return {"folds": int(len(a)), "folds_auc_above_0.5": k, "mean_auc": float(a.mean()),
            "p_value": float(stats.binomtest(k, len(a), 0.5, alternative="greater").pvalue) if len(a) else float("nan")}


def yearly_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in frame.groupby(frame.index.year):
        y = g["target_direction"].to_numpy().astype(int)
        rows.append({"year": int(year), "days": len(g), "accuracy": float(np.mean((g["raw_prob"] > 0.5) == y)),
                     "always_up_accuracy": float(y.mean()),
                     "auc": float(roc_auc_score(y, g["raw_prob"])) if len(np.unique(y)) == 2 else float("nan")})
    return pd.DataFrame(rows).set_index("year")


def run_walk_forward(
    cfg: Dict[str, Any], ticker: Optional[str] = None, initial_days: Optional[int] = None,
    test_days: Optional[int] = None, make_plots: bool = True, save: bool = True,
) -> Dict[str, Any]:
    wc, xc = merged_wf_config(cfg), merged_xgb_config(cfg)
    if initial_days:
        wc["initial_train_days"] = initial_days
    if test_days:
        wc["test_days"] = test_days
    ticker, full, sup = load_supervised(cfg, ticker)
    cols = feature_columns(sup, scale_free_only=(wc["feature_set"] == "scale_free"))

    print(f"\n=== {ticker}: walk-forward validation (XGBoost, default parameters) ===")
    print(f"{len(sup)} usable days | first training window {wc['initial_train_days']} days | "
          f"test blocks of {wc['test_days']} days | {len(cols)} features ({wc['feature_set']})")
    preds, folds = walk_forward_predictions(
        sup, cols, xc["params"], xc["n_estimators"], xc["early_stopping_rounds"], xc["seed"], xc["use_class_weight"],
        wc["initial_train_days"], wc["test_days"], wc["gap"], wc["val_frac"])
    frame = sup.loc[preds.index].join(preds)
    print(f"{len(folds)} folds -> {len(frame)} out-of-sample days: {frame.index.min().date()} to {frame.index.max().date()}")

    print("\n=== Per-fold results ===")
    print(fmt(folds[["test_start", "test_end", "n_test", "auc", "accuracy", "always_up_accuracy", "clf_best_iter"]]))
    sign = fold_sign_test(folds["auc"])
    print(f"\nAUC above 0.5 in {sign['folds_auc_above_0.5']} of {sign['folds']} folds | mean fold AUC "
          f"{sign['mean_auc']:.3f} | sign-test p = {sign['p_value']:.3f}")

    years = yearly_table(frame)
    print("\n=== By calendar year (pooled out-of-sample days) ===")
    print(fmt(years))

    cls_df, _cms = tables([
        classification_row("always_up", frame, np.ones(len(frame)), 1, False),
        classification_row("momentum", frame, frame["return"].to_numpy(), 1, False),
        classification_row("xgb_walk_forward", frame, frame["raw_prob"].to_numpy(), 1, True),
    ])
    print("\n=== Pooled out-of-sample classification ===")
    print(fmt(cls_df))
    reg_df = pd.DataFrame([regression_row("previous_day", frame, np.zeros(len(frame))),
                           regression_row("xgb_walk_forward", frame, frame["pred_return"].to_numpy())]).set_index("model")
    reg_df["RMSE_vs_naive"] = reg_df["RMSE"] / reg_df.loc["previous_day", "RMSE"]
    print("\n=== Pooled out-of-sample regression ===")
    print(fmt(reg_df))

    # ---- pooled long/flat backtest ----
    bcfg = cfg.get("backtest") or {}
    ic, cost = float(bcfg.get("initial_capital", 100000)), float(bcfg.get("cost_bps", 5))
    execution = bcfg.get("execution", "next_open")
    market = build_market_data(full, frame.index)
    strategies = {"buy_and_hold": buy_and_hold(len(frame)),
                  "xgb_wf_clf": positions_from_prob(frame["raw_prob"].to_numpy()),
                  "xgb_wf_reg": positions_from_return(frame["pred_return"].to_numpy())}
    curves = {n: run_backtest(market, p, ic, cost, execution) for n, p in strategies.items()}
    metrics = pd.DataFrame({n: performance_metrics(c, ic) for n, c in curves.items()}).T
    print(f"\n=== Pooled walk-forward backtest (${ic:,.0f}, {cost} bps, {execution}) ===")
    show = metrics[["total_return_%", "CAGR_%", "sharpe", "max_drawdown_%", "n_trades", "exposure_%", "cost_drag_%"]]
    print(show.round(2).to_string())
    null = {}
    for n in ("xgb_wf_clf", "xgb_wf_reg"):
        sh, _ = random_null(market, float(np.mean(np.abs(strategies[n]))), 1000, cost, execution, False, 42)
        null[n] = {"sharpe": metrics.loc[n, "sharpe"], **null_summary(metrics.loc[n, "sharpe"], sh)}
    null_df = pd.DataFrame(null).T
    null_df["p_bonferroni"] = (null_df["p_value"] * len(null_df)).clip(upper=1.0)
    print("\nBetter than luck? (1000 random strategies with the same exposure)")
    print(null_df.round(3).to_string())
    print("\nExperimental research, not financial advice. Walk-forward is a stronger test than one split, but it is "
          "still one asset, with simplified costs.")

    results = {"ticker": ticker, "config": wc, "n_folds": int(len(folds)),
               "oos_period": [str(frame.index.min().date()), str(frame.index.max().date())],
               "fold_sign_test": sign, "folds": folds.reset_index().to_dict("records"),
               "yearly": years.reset_index().to_dict("records"),
               "classification": cls_df.reset_index().to_dict("records"),
               "regression": reg_df.reset_index().to_dict("records"),
               "backtest": metrics.reset_index().to_dict("records"), "null": null_df.reset_index().to_dict("records")}
    figures = []
    if make_plots:
        fig_dir = Path(resolve_path(cfg["paths"]["figures_dir"]))
        figures.append(mp.plot_fold_auc(folds, fig_dir / f"{ticker}_wf_fold_auc.png"))
        figures.append(mp.plot_yearly_accuracy(years, fig_dir / f"{ticker}_wf_yearly_accuracy.png"))
        figures.append(bp.plot_equity(curves, f"{ticker}: walk-forward out-of-sample portfolio value ({cost} bps)",
                                      fig_dir / f"{ticker}_wf_equity.png"))
        print("\nSaved figures:\n  " + "\n  ".join(figures))
    if save:
        reports = Path(resolve_path(cfg["paths"].get("reports_dir", "reports")))
        (reports / f"walkforward_results_{ticker}.json").write_text(json.dumps(results, indent=2, default=json_default))
        frame[["fold", "raw_prob", "pred_return"]].to_csv(reports / f"walkforward_predictions_{ticker}.csv", index_label="Date")
        print(f"\nSaved walk-forward results to {reports}")
    results["figures"] = figures
    return results
