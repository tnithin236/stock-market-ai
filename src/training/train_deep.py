"""Generic runner for sequence models (LSTM, Transformer) with one honest protocol.

  * Scaler is fitted on TRAIN rows only; windows only read past features.
  * Early stopping, LR scheduling, checkpoint choice and tuning use VALIDATION.
  * TEST is evaluated once at the end and never used to choose anything.
"""
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import resolve_path
from src.evaluation.permutation import permutation_importance
from src.evaluation.reporting import classification_row, fmt, json_default, regression_row, tables
from src.features.engineering import feature_columns
from src.models.arch_config import ARCH_HP_KEYS, ARCH_LABEL, merged_deep_config
from src.models.baselines import MajorityClassBaseline, MomentumBaseline
from src.models.calibration import PlattCalibrator
from src.training.dataset import load_supervised, split_dataset
from src.training.lstm_fit import fit_task
from src.training.nn_trainer import TrainConfig, get_device, predict_array
from src.training.sequences import prepare_sequence_data
from src.training.tuning_deep import tune_deep
from src.visualization import model_plots as mp

logger = logging.getLogger(__name__)

MIN_TUNING_GAIN = 0.001  # tuned hyperparameters must beat the default validation loss by >= 0.1%


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def run_deep(
    arch: str,
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
    lc = merged_deep_config(cfg, arch)
    label = ARCH_LABEL[arch]
    ticker, _full, sup = load_supervised(cfg, ticker, start, end, refresh)
    split = split_dataset(sup, cfg)
    cols = feature_columns(sup, scale_free_only=(lc["feature_set"] == "scale_free"))
    scaling_method = cfg.get("scaling", {}).get("method", "standard")
    device, seed = get_device(), lc["seed"]

    y_std = float(split.train["target_return"].std())
    n_up = int(split.train["target_direction"].sum())
    pos_weight = ((len(split.train) - n_up) / n_up) if lc["use_class_weight"] else None

    base_tc = TrainConfig(
        lr=lc["lr"], weight_decay=lc["weight_decay"], batch_size=lc["batch_size"],
        max_epochs=lc["max_epochs"], patience=lc["patience"], lr_patience=lc["lr_patience"],
        lr_factor=lc["lr_factor"], grad_clip=lc["grad_clip"])
    default_hp = {k: lc[k] for k in ARCH_HP_KEYS[arch]}

    cache: Dict[int, Any] = {}

    def build(seq_len: int):
        if seq_len not in cache:
            cache[seq_len] = prepare_sequence_data(sup, split, cols, seq_len, scaling_method, lc["clip_scaled"])
        return cache[seq_len]

    models_dir = Path(resolve_path(cfg["paths"]["models_dir"]))
    print(f"\n=== {ticker}: {label} (device: {device}) ===")
    print(split.summary().to_string())
    print(f"Inputs: {len(cols)} features ({lc['feature_set']}) | seq_len={lc['seq_len']} | "
          f"target return std (train)={y_std:.4f} | pos_weight={None if pos_weight is None else round(pos_weight, 3)}")
    print(f"Hyperparameters: {default_hp}")

    # ---- 1) Train (and optionally tune on VALIDATION) ----
    final: Dict[str, Any] = {}
    tuning_info: Dict[str, Any] = {}
    for task in ("regression", "classification"):
        ckpt = models_dir / f"{ticker}_{arch}_{task}_best.pt" if save else None
        hp = dict(default_hp)
        data, scaler = build(hp["seq_len"])
        print(f"\nTraining {task} {label} ...")
        model, res = fit_task(task, data, hp, base_tc, device, y_std, pos_weight, seed, ckpt, arch=arch)
        default_val = res["best_val_loss"]
        print(f"  best epoch {res['best_epoch']} of {res['epochs_run']} run | best validation loss {default_val:.6f}")

        if tune:
            trials = n_trials or lc["tuning"]["n_trials"]
            print(f"  tuning {task} with Optuna ({trials} trials, validation loss) ... this takes a few minutes")
            best_hp, best_val, n_run = tune_deep(
                arch, task, build, base_tc, device, y_std, pos_weight, seed, trials,
                lc["tuning"]["seq_len_choices"], lc["tuning"]["max_epochs"])
            adopted = best_val < default_val * (1 - MIN_TUNING_GAIN)
            tuning_info[task] = {"default_val_loss": default_val, "tuned_val_loss": best_val, "trials": n_run,
                                 "best_params": best_hp, "adopted": adopted}
            if adopted:
                hp = {**default_hp, **best_hp}
                data, scaler = build(hp["seq_len"])
                model, res = fit_task(task, data, hp, base_tc, device, y_std, pos_weight, seed, ckpt, arch=arch)
                print(f"  validation loss: default {default_val:.6f} -> tuned {res['best_val_loss']:.6f} "
                      "(part of this gain may be overfitting to validation; judge on TEST)")
            else:
                print(f"  tuned loss {best_val:.6f} did not beat default {default_val:.6f} by at least "
                      f"{MIN_TUNING_GAIN:.1%}; keeping default hyperparameters")
        final[task] = {"model": model, "res": res, "hp": hp, "data": data, "scaler": scaler}

    # ---- 2) Predictions, calibration, evaluation ----
    reg, clf = final["regression"], final["classification"]
    raw_val_logits = predict_array(clf["model"], clf["data"]["val"].X, device)
    calibrator = PlattCalibrator().fit(_sigmoid(raw_val_logits), clf["data"]["val"].y_dir)

    results: Dict[str, Any] = {
        "ticker": ticker, "architecture": arch, "feature_set": lc["feature_set"], "n_features": len(cols),
        "hyperparameters": {"regression": reg["hp"], "classification": clf["hp"]},
        "tuning": tuning_info, "regression": {}, "classification": {},
        "training": {t: {"best_epoch": final[t]["res"]["best_epoch"], "epochs_run": final[t]["res"]["epochs_run"],
                         "best_val_loss": final[t]["res"]["best_val_loss"],
                         "history": final[t]["res"]["history"]} for t in final},
    }
    majority = MajorityClassBaseline().fit(split.train["target_direction"])
    momentum = MomentumBaseline()
    cache_eval: Dict[str, Any] = {}

    for name in ("val", "test"):
        reg_part = sup.loc[reg["data"][name].index]
        pred_ret = predict_array(reg["model"], reg["data"][name].X, device) * y_std
        reg_df = pd.DataFrame([regression_row("previous_day", reg_part, np.zeros(len(reg_part))),
                               regression_row(f"{arch}_reg", reg_part, pred_ret)]).set_index("model")
        reg_df["RMSE_vs_naive"] = reg_df["RMSE"] / reg_df.loc["previous_day", "RMSE"]

        clf_part = sup.loc[clf["data"][name].index]
        raw = _sigmoid(predict_array(clf["model"], clf["data"][name].X, device))
        cal = calibrator.predict(raw)
        cls_df, cms = tables([
            classification_row("always_majority", clf_part, majority.score(clf_part), majority.label, False),
            classification_row("momentum", clf_part, momentum.score(clf_part), majority.label, False),
            classification_row(f"{arch}_raw", clf_part, raw, majority.label, True),
            classification_row(f"{arch}_calibrated", clf_part, cal, majority.label, True),
        ])
        print(f"\n=== Regression: next-day close ({name.upper()}) ===")
        print(fmt(reg_df))
        print(f"\n=== Classification: next-day direction ({name.upper()}) ===")
        print(fmt(cls_df))
        for m in (f"{arch}_raw", f"{arch}_calibrated"):
            print(f"  confusion [[TN, FP], [FN, TP]] {m}: {cms[m]}")
        results["regression"][name] = reg_df.reset_index().to_dict("records")
        results["classification"][name] = cls_df.reset_index().to_dict("records")
        cache_eval[name] = {"reg_part": reg_part, "pred_ret": pred_ret, "clf_part": clf_part,
                            "raw": raw, "cal": cal, "cms": cms}

    print("\nHow to read this:")
    print(" - Regression: RMSE_vs_naive < 1 and oosR2_ret > 0 on TEST would mean a real edge over 'tomorrow = today'.")
    print(" - Classification: needs accuracy above always_majority with a small p_vs_majority AND an AUC interval above 0.5.")
    print(" - A calibrated row that is constant means the raw score had no usable (positive) signal on validation.")

    # ---- 3) Permutation importance on TEST (explanation only; no training) ----
    reps = lc["permutation"]["n_repeats"]
    test_clf, test_reg = clf["data"]["test"], reg["data"]["test"]
    imp_clf = permutation_importance(
        lambda X: predict_array(clf["model"], X, device), test_clf.X, test_clf.y_dir,
        lambda y, s: roc_auc_score(y, s), reps, seed, cols)
    imp_reg = permutation_importance(
        lambda X: predict_array(reg["model"], X, device), test_reg.X, test_reg.y_ret / y_std,
        lambda y, s: -float(np.sqrt(np.mean((s - y) ** 2))), reps, seed, cols)
    for lab, imp, unit in (("classifier", imp_clf, "AUC drop"), ("regressor", imp_reg, "RMSE increase (scaled)")):
        print(f"\n=== Permutation importance: {lab} (test; {unit}; top 10) ===")
        print(fmt(imp.head(10), 5))
    print("\nCaution: values within one std of zero mean no reliable dependence. Correlated features split credit "
          "and permuting creates unrealistic inputs, so treat this as a rough guide.")
    results["permutation_importance"] = {"classifier": imp_clf.head(15).reset_index().to_dict("records"),
                                         "regressor": imp_reg.head(15).reset_index().to_dict("records")}

    # ---- 4) Plots ----
    figures = []
    if make_plots:
        fig_dir = Path(resolve_path(cfg["paths"]["figures_dir"]))
        for task, blob in final.items():
            figures.append(mp.plot_training_curves(
                blob["res"]["history"], f"{ticker} {label} {task}: train vs validation loss",
                fig_dir / f"{ticker}_{arch}_{task}_curves.png", blob["res"]["best_epoch"]))
        te = cache_eval["test"]
        rp = te["reg_part"]
        figures.append(mp.plot_actual_vs_predicted(
            rp.index, rp["target_close"].to_numpy(),
            {"previous day": rp["Close"].to_numpy(), label: rp["Close"].to_numpy() * (1 + te["pred_ret"])},
            f"{ticker}: actual vs predicted next-day close (last 120 test days)",
            fig_dir / f"{ticker}_{arch}_actual_vs_pred.png"))
        figures.append(mp.plot_confusion(te["cms"][f"{arch}_calibrated"], f"{ticker} {label} (calibrated) - test",
                                         fig_dir / f"{ticker}_{arch}_confusion.png"))
        y_te = te["clf_part"]["target_direction"].to_numpy().astype(int)
        figures.append(mp.plot_roc(y_te, {label: te["raw"], "momentum": momentum.score(te["clf_part"])},
                                   f"{ticker} {label} ROC - test", fig_dir / f"{ticker}_{arch}_roc.png"))
        figures.append(mp.plot_calibration(y_te, {"raw": te["raw"], "Platt-calibrated": te["cal"]},
                                           f"{ticker} {label} calibration - test",
                                           fig_dir / f"{ticker}_{arch}_calibration.png"))
        figures.append(mp.plot_importance(imp_clf["importance_mean"],
                                          f"{ticker} {label} classifier: permutation importance",
                                          fig_dir / f"{ticker}_{arch}_permutation.png",
                                          xlabel="AUC drop when shuffled"))
        print("\nSaved figures:\n  " + "\n  ".join(figures))
    results["figures"] = figures

    # ---- 5) Persist artifacts (tensors + plain dicts only, so torch.load(weights_only=True) works) ----
    if save:
        import torch

        meta = {"ticker": ticker, "architecture": arch, "trained_on": date.today().isoformat(),
                "torch_version": torch.__version__,
                "train_period": [str(split.train.index.min().date()), str(split.train.index.max().date())],
                "val_period": [str(split.val.index.min().date()), str(split.val.index.max().date())]}
        for task, blob in final.items():
            torch.save({"task": task, "architecture": arch, "state_dict": blob["model"].state_dict(),
                        "hparams": blob["hp"], "features": list(cols), "y_std": y_std, "meta": meta},
                       models_dir / f"{ticker}_{arch}_{task}.pt")
        reg["scaler"].save(models_dir / f"{ticker}_{arch}_scaler.joblib")
        joblib.dump(calibrator, models_dir / f"{ticker}_{arch}_calibrator.joblib")
        reports = Path(resolve_path(cfg["paths"].get("reports_dir", "reports")))
        out = reports / f"{arch}_results_{ticker}.json"
        out.write_text(json.dumps(results, indent=2, default=json_default))
        print(f"\nSaved models to {models_dir}\nSaved results to {out}")
    return results
