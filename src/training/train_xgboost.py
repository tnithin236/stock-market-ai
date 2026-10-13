"""Step 3: XGBoost regression + classification with honest evaluation.

Protocol (read this before trusting any number):
  * Train on TRAIN. Early stopping, feature-set choice, tuning and calibration use VALIDATION.
  * TEST is evaluated once, at the end, and is never used to make a choice.
  * Validation numbers are slightly optimistic (we selected on them). Test numbers are the honest ones.
"""
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.config import resolve_path
from src.evaluation.explain import gain_importance, mean_abs_contribution, tree_contributions
from src.evaluation.metrics import classification_metrics, directional_accuracy, regression_metrics
from src.evaluation.stats import (accuracy_vs_rate_pvalue, bootstrap_auc_ci, r2_vs_zero,
                                  wilson_interval)
from src.features.engineering import feature_columns
from src.models.baselines import MajorityClassBaseline, MomentumBaseline
from src.models.xgb_models import (TARGETS, PlattCalibrator, fit_model, make_model,
                                   merged_xgb_config, predict_proba_up, predict_return,
                                   return_to_price, save_bundle)
from src.training.dataset import load_supervised, split_dataset
from src.training.tuning import tune_xgboost
from src.visualization import model_plots as mp

logger = logging.getLogger(__name__)


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def _fmt(df: pd.DataFrame, digits: int = 4) -> str:
    return df.round(digits).to_string()


# ---------------------------------------------------------------- evaluation rows
def _regression_row(name: str, part: pd.DataFrame, pred_ret: np.ndarray) -> Dict[str, Any]:
    close = part["Close"].to_numpy()
    y_close = part["target_close"].to_numpy()
    y_ret = part["target_return"].to_numpy()
    pred_price = return_to_price(close, pred_ret)
    row: Dict[str, Any] = {"model": name, **regression_metrics(y_close, pred_price)}
    row["dir_acc_%"] = directional_accuracy(close, y_close, pred_price) * 100
    row["RMSE_ret"] = float(np.sqrt(np.mean((pred_ret - y_ret) ** 2)))
    row["oosR2_ret"] = r2_vs_zero(y_ret, pred_ret)
    return row


def _classification_row(name: str, part: pd.DataFrame, score: np.ndarray, majority_label: int,
                        probabilistic: bool, threshold: float = 0.5) -> Dict[str, Any]:
    y = part["target_direction"].to_numpy().astype(int)
    pred = (score >= threshold).astype(int) if probabilistic else (score > 0).astype(int)
    met = classification_metrics(y, pred, score)
    cm = met.pop("confusion_matrix")
    n, k = len(y), int((pred == y).sum())
    lo, hi = wilson_interval(k, n)
    p_major = float(np.mean(y == majority_label))
    row: Dict[str, Any] = {
        "model": name,
        "accuracy": met["accuracy"],
        "acc_CI95": f"[{lo:.3f}, {hi:.3f}]",
        "p_vs_majority": accuracy_vs_rate_pvalue(k, n, p_major),
        "precision": met["precision"], "recall": met["recall"], "f1": met["f1"],
        "roc_auc": met["roc_auc"], "balanced_acc": met["balanced_accuracy"], "mcc": met["mcc"],
    }
    if met["roc_auc"] is not None:
        a, b = bootstrap_auc_ci(y, score)
        row["auc_CI95"] = f"[{a:.3f}, {b:.3f}]"
    if probabilistic:
        clipped = np.clip(score, 1e-6, 1 - 1e-6)
        row["brier"] = float(brier_score_loss(y, clipped))
        row["logloss"] = float(log_loss(y, clipped, labels=[0, 1]))
    row["_cm"] = cm
    return row


def _tables(rows):
    cms = {r["model"]: r.pop("_cm") for r in rows}
    return pd.DataFrame(rows).set_index("model"), cms


# ---------------------------------------------------------------- main routine
def run_xgboost(
    cfg: Dict[str, Any],
    ticker: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    refresh: bool = False,
    tune: bool = False,
    n_trials: Optional[int] = None,
    make_plots: bool = True,
    save: bool = True,
) -> Dict[str, Any]:
    xc = merged_xgb_config(cfg)
    ticker, _full, sup = load_supervised(cfg, ticker, start, end, refresh)
    split = split_dataset(sup, cfg)
    tr, va, te = split.train, split.val, split.test
    seed = xc["seed"]

    n_up = int(tr["target_direction"].sum())
    n_down = len(tr) - n_up
    spw = (n_down / n_up) if xc["use_class_weight"] else None

    feature_sets = {
        "all": feature_columns(sup),
        "scale_free": feature_columns(sup, scale_free_only=True),
    }

    print(f"\n=== {ticker}: XGBoost ===")
    print(split.summary().to_string())
    print(f"Features: all={len(feature_sets['all'])}, scale_free={len(feature_sets['scale_free'])} | "
          f"class weight (scale_pos_weight)={spw if spw is None else round(spw, 3)}")

    # ---- 1) Feature-set comparison on VALIDATION (default params) ----
    comp, fitted = [], {}
    for task in ("regression", "classification"):
        target = TARGETS[task]
        for fs, cols in feature_sets.items():
            model = make_model(task, xc["params"], xc["n_estimators"], xc["early_stopping_rounds"],
                               seed, spw if task == "classification" else None)
            fit_model(model, tr[cols], tr[target], va[cols], va[target])
            fitted[(task, fs)] = model
            row: Dict[str, Any] = {"task": task, "features": fs, "n_features": len(cols),
                                   "best_iter": int(model.best_iteration)}
            if task == "regression":
                pr = predict_return(model, va[cols])
                y = va[target].to_numpy()
                row["val_RMSE_ret"] = float(np.sqrt(np.mean((pr - y) ** 2)))
                row["val_RMSE_zero"] = float(np.sqrt(np.mean(y ** 2)))
                row["select_by"] = row["val_RMSE_ret"]
            else:
                p = predict_proba_up(model, va[cols])
                y = va[target].to_numpy().astype(int)
                row["val_logloss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
                row["val_auc"] = float(roc_auc_score(y, p))
                row["val_acc"] = float(np.mean((p >= 0.5) == y))
                row["select_by"] = row["val_logloss"]
            comp.append(row)

    comp_df = pd.DataFrame(comp)
    print("\n=== Feature-set comparison (default params, VALIDATION only) ===")
    print(_fmt(comp_df.drop(columns=["select_by"]).set_index(["task", "features"])))

    chosen = {}
    for task in ("regression", "classification"):
        sub = comp_df[comp_df["task"] == task]
        chosen[task] = sub.loc[sub["select_by"].idxmin(), "features"]
    print(f"Chosen by validation loss -> regression: {chosen['regression']}, "
          f"classification: {chosen['classification']}")

    # ---- 2) Optional tuning (VALIDATION only) ----
    final, params_used, tuning_info = {}, {}, {}
    for task in ("regression", "classification"):
        cols = feature_sets[chosen[task]]
        target = TARGETS[task]
        model = fitted[(task, chosen[task])]
        params = dict(xc["params"])
        if tune:
            trials = n_trials or xc["tuning"]["n_trials"]
            print(f"\nTuning {task} with Optuna ({trials} trials, validation loss)...")
            best_params, best_val, n_run = tune_xgboost(
                task, tr[cols], tr[target], va[cols], va[target], trials, seed,
                spw if task == "classification" else None,
                xc["n_estimators"], xc["early_stopping_rounds"])
            default_val = float(model.best_score)
            adopted = best_val < default_val
            if adopted:
                model = make_model(task, best_params, xc["n_estimators"], xc["early_stopping_rounds"],
                                   seed, spw if task == "classification" else None)
                fit_model(model, tr[cols], tr[target], va[cols], va[target])
                params = best_params
            tuning_info[task] = {"default_val_loss": default_val, "tuned_val_loss": best_val,
                                 "trials": n_run, "best_params": best_params, "adopted": adopted}
            if adopted:
                print(f"  validation loss: default {default_val:.6f} -> tuned {best_val:.6f} "
                      "(part of this gain may be overfitting to validation; judge on TEST)")
            else:
                print(f"  tuned loss {best_val:.6f} did not beat default {default_val:.6f}; keeping default params")
        final[task], params_used[task] = model, params

    reg, clf = final["regression"], final["classification"]
    reg_cols, clf_cols = feature_sets[chosen["regression"]], feature_sets[chosen["classification"]]

    # ---- 3) Calibration (fit on VALIDATION raw probabilities) ----
    raw_val = predict_proba_up(clf, va[clf_cols])
    calibrator = PlattCalibrator().fit(raw_val, va["target_direction"])

    # ---- 4) Final evaluation: validation and TEST ----
    majority = MajorityClassBaseline().fit(tr["target_direction"])
    momentum = MomentumBaseline()
    results: Dict[str, Any] = {
        "ticker": ticker, "chosen_features": chosen, "params": params_used, "tuning": tuning_info,
        "feature_set_comparison": comp_df.to_dict("records"), "regression": {}, "classification": {},
    }
    eval_cache = {}
    for name, part in (("val", va), ("test", te)):
        pr = predict_return(reg, part[reg_cols])
        reg_rows = [_regression_row("previous_day", part, np.zeros(len(part))),
                    _regression_row(f"xgb_reg[{chosen['regression']}]", part, pr)]
        reg_df = pd.DataFrame(reg_rows).set_index("model")
        reg_df["RMSE_vs_naive"] = reg_df["RMSE"] / reg_df.loc["previous_day", "RMSE"]

        raw = predict_proba_up(clf, part[clf_cols])
        cal = calibrator.predict(raw)
        cls_rows = [
            _classification_row("always_majority", part, majority.score(part), majority.label, False),
            _classification_row("momentum", part, momentum.score(part), majority.label, False),
            _classification_row("xgb_raw", part, raw, majority.label, True),
            _classification_row("xgb_calibrated", part, cal, majority.label, True),
        ]
        cls_df, cms = _tables(cls_rows)

        print(f"\n=== Regression: next-day close ({name.upper()}) ===")
        print(_fmt(reg_df))
        print(f"\n=== Classification: next-day direction ({name.upper()}) ===")
        print(_fmt(cls_df))
        for mname in ("xgb_raw", "xgb_calibrated"):
            print(f"  confusion [[TN, FP], [FN, TP]] {mname}: {cms[mname]}")

        results["regression"][name] = reg_df.reset_index().to_dict("records")
        results["classification"][name] = cls_df.reset_index().to_dict("records")
        eval_cache[name] = {"pred_ret": pr, "raw": raw, "cal": cal, "cms": cms}

    print("\nHow to read this:")
    print(" - Regression: RMSE_vs_naive < 1 and oosR2_ret > 0 mean the model beats 'tomorrow = today'.")
    print(" - Classification: it only counts if accuracy beats always_majority with a small p_vs_majority "
          "AND roc_auc is clearly above 0.5 on BOTH validation and test.")
    print(" - Confidence intervals are wide at ~436 samples; 1-2 point differences are noise.")

    # ---- 5) Explainability (on the TEST period; no training involved) ----
    explain: Dict[str, Any] = {}
    contrib_clf = contrib_reg = None
    try:
        contrib_clf, _ = tree_contributions(clf, te[clf_cols])
        contrib_reg, _ = tree_contributions(reg, te[reg_cols])
    except Exception as exc:
        logger.warning("SHAP contributions unavailable (%s)", exc)

    for label, model, cols, contrib in (("classifier", clf, clf_cols, contrib_clf),
                                        ("regressor", reg, reg_cols, contrib_reg)):
        gain = gain_importance(model, cols)
        table = pd.DataFrame({"gain_share": gain.head(10)})
        if contrib is not None:
            shap_rank = mean_abs_contribution(contrib)
            table["mean_abs_SHAP"] = shap_rank.reindex(table.index)
            explain[label] = {"gain": gain.head(15).to_dict(), "mean_abs_shap": shap_rank.head(15).to_dict()}
        else:
            explain[label] = {"gain": gain.head(15).to_dict()}
        print(f"\n=== Top 10 features: {label} (gain share; mean |SHAP| on test) ===")
        print(_fmt(table, 5))
    results["explain"] = explain
    print("\nCaution: importance shows what the model USED, not what is causally true. "
          "In a near-random target, top features are often noise that happened to split well.")

    # ---- 6) Plots ----
    figures = []
    if make_plots:
        fig_dir = Path(resolve_path(cfg["paths"]["figures_dir"]))
        figures.append(mp.plot_importance(gain_importance(clf, clf_cols), f"{ticker} classifier: gain importance",
                                          fig_dir / f"{ticker}_xgb_clf_importance.png", xlabel="share of gain"))
        figures.append(mp.plot_importance(gain_importance(reg, reg_cols), f"{ticker} regressor: gain importance",
                                          fig_dir / f"{ticker}_xgb_reg_importance.png", xlabel="share of gain"))
        if contrib_clf is not None:
            figures.append(mp.plot_importance(mean_abs_contribution(contrib_clf),
                                              f"{ticker} classifier: mean |SHAP| (test)",
                                              fig_dir / f"{ticker}_xgb_clf_shap_bar.png", xlabel="mean |SHAP| (log-odds)"))
            beeswarm = mp.plot_shap_summary(contrib_clf, te[clf_cols], fig_dir / f"{ticker}_xgb_clf_shap_summary.png")
            if beeswarm:
                figures.append(beeswarm)
        te_pred_price = return_to_price(te["Close"], eval_cache["test"]["pred_ret"])
        figures.append(mp.plot_actual_vs_predicted(
            te.index, te["target_close"].to_numpy(),
            {"previous day": te["Close"].to_numpy(), "XGBoost": te_pred_price},
            f"{ticker}: actual vs predicted next-day close (last 120 test days)",
            fig_dir / f"{ticker}_xgb_actual_vs_pred.png"))
        figures.append(mp.plot_confusion(eval_cache["test"]["cms"]["xgb_calibrated"],
                                         f"{ticker} XGBoost (calibrated) - test",
                                         fig_dir / f"{ticker}_xgb_confusion.png"))
        y_te = te["target_direction"].to_numpy().astype(int)
        figures.append(mp.plot_roc(y_te, {"XGBoost": eval_cache["test"]["raw"], "momentum": momentum.score(te)},
                                   f"{ticker} ROC - test", fig_dir / f"{ticker}_xgb_roc.png"))
        figures.append(mp.plot_calibration(y_te, {"raw": eval_cache["test"]["raw"],
                                                  "Platt-calibrated": eval_cache["test"]["cal"]},
                                           f"{ticker} calibration - test", fig_dir / f"{ticker}_xgb_calibration.png"))
        print("\nSaved figures:\n  " + "\n  ".join(figures))
    results["figures"] = figures

    # ---- 7) Persist artifacts ----
    if save:
        import xgboost

        models_dir = Path(resolve_path(cfg["paths"]["models_dir"]))
        meta_base = {
            "ticker": ticker, "trained_on": date.today().isoformat(),
            "train_period": [str(tr.index.min().date()), str(tr.index.max().date())],
            "val_period": [str(va.index.min().date()), str(va.index.max().date())],
            "xgboost_version": xgboost.__version__,
        }
        save_bundle(models_dir / f"{ticker}_xgb_regression.joblib", reg, "regression", reg_cols, None,
                    {**meta_base, "feature_set": chosen["regression"], "params": params_used["regression"],
                     "best_iteration": int(reg.best_iteration), "target": "next-day return"})
        save_bundle(models_dir / f"{ticker}_xgb_classification.joblib", clf, "classification", clf_cols, calibrator,
                    {**meta_base, "feature_set": chosen["classification"], "params": params_used["classification"],
                     "best_iteration": int(clf.best_iteration), "calibration": "platt (validation)",
                     "target": "next-day direction"})
        reports = Path(resolve_path(cfg["paths"].get("reports_dir", "reports")))
        out = reports / f"xgboost_results_{ticker}.json"
        out.write_text(json.dumps(results, indent=2, default=_json_default))
        print(f"\nSaved models to {models_dir}\nSaved results to {out}")
    return results
