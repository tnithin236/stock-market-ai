"""Evaluation rows/tables shared by every model so results are directly comparable."""
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.evaluation.metrics import classification_metrics, directional_accuracy, regression_metrics
from src.evaluation.stats import (accuracy_vs_rate_pvalue, bootstrap_auc_ci, r2_vs_zero,
                                  wilson_interval)


def json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def fmt(df: pd.DataFrame, digits: int = 4) -> str:
    return df.round(digits).to_string()


def regression_row(name: str, part: pd.DataFrame, pred_ret: np.ndarray) -> Dict[str, Any]:
    """Score a next-day-return forecast on price level and on returns."""
    close = part["Close"].to_numpy()
    y_close = part["target_close"].to_numpy()
    y_ret = part["target_return"].to_numpy()
    pred_price = close * (1.0 + np.asarray(pred_ret, dtype=float))
    row: Dict[str, Any] = {"model": name, **regression_metrics(y_close, pred_price)}
    row["dir_acc_%"] = directional_accuracy(close, y_close, pred_price) * 100
    row["RMSE_ret"] = float(np.sqrt(np.mean((pred_ret - y_ret) ** 2)))
    row["oosR2_ret"] = r2_vs_zero(y_ret, pred_ret)
    return row


def classification_row(name: str, part: pd.DataFrame, score: np.ndarray, majority_label: int,
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


def tables(rows):
    """Split rows into (metrics DataFrame, {model: confusion matrix})."""
    cms = {r["model"]: r.pop("_cm") for r in rows}
    return pd.DataFrame(rows).set_index("model"), cms
