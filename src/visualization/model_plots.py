"""Static (PNG) model diagnostics via matplotlib."""
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

logger = logging.getLogger(__name__)


def _save(fig, path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_importance(series: pd.Series, title: str, path, top: int = 15, xlabel: str = "") -> str:
    s = series.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(s) + 1.5))
    ax.barh(s.index, s.values, color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return _save(fig, path)


def plot_shap_summary(contrib: pd.DataFrame, X: pd.DataFrame, path, max_display: int = 15) -> Optional[str]:
    """Beeswarm via the shap package. Optional: failure only skips this one figure."""
    try:
        import shap

        plt.figure()
        shap.summary_plot(contrib.values, X, show=False, max_display=max_display)
        fig = plt.gcf()
        return _save(fig, path)
    except Exception as exc:  # pragma: no cover - depends on shap/matplotlib versions
        logger.warning("Skipping SHAP beeswarm plot (%s)", exc)
        plt.close("all")
        return None


def plot_actual_vs_predicted(dates, actual, predictions: Dict[str, np.ndarray], title: str, path,
                             n_last: int = 120) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates[-n_last:], np.asarray(actual)[-n_last:], label="Actual close", color="black", lw=1.8)
    for name, vals in predictions.items():
        ax.plot(dates[-n_last:], np.asarray(vals)[-n_last:], label=name, lw=1.2, alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, path)


def plot_confusion(cm, title: str, path) -> str:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred DOWN", "Pred UP"])
    ax.set_yticks([0, 1], ["Actual DOWN", "Actual UP"])
    ax.set_title(title)
    return _save(fig, path)


def plot_roc(y_true, scores: Dict[str, np.ndarray], title: str, path) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_calibration(y_true, probs: Dict[str, np.ndarray], title: str, path, n_bins: int = 8) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in probs.items():
        frac, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed share of UP days")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_training_curves(history: Dict[str, list], title: str, path, best_epoch: Optional[int] = None) -> str:
    """Train vs validation loss per epoch (the overfitting diagnostic)."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(epochs, history["train_loss"], label="train loss")
    ax.plot(epochs, history["val_loss"], label="validation loss")
    if best_epoch:
        ax.axvline(best_epoch, color="grey", ls="--", lw=1, label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_model_comparison(cls: pd.DataFrame, path, title: str = "Test-set model comparison") -> str:
    """Accuracy and ROC-AUC with 95% intervals; the dotted line is the bar to beat."""
    df = cls.copy()
    df = df[df["roc_auc"].notna()] if "roc_auc" in df else df
    names = list(df.index)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(names) + 2.2), sharey=True)
    y = np.arange(len(names))
    for ax, col, lo, hi, ref, ttl in (
        (axes[0], "accuracy", "acc_lo", "acc_hi", float(df.loc["always_majority", "accuracy"]) if "always_majority" in df.index else 0.5, "Accuracy"),
        (axes[1], "roc_auc", "auc_lo", "auc_hi", 0.5, "ROC-AUC"),
    ):
        vals = df[col].to_numpy(dtype=float)
        err = np.vstack([vals - df[lo].to_numpy(dtype=float), df[hi].to_numpy(dtype=float) - vals])
        err = np.nan_to_num(err, nan=0.0)
        ax.errorbar(vals, y, xerr=err, fmt="o", color="#3b82f6", ecolor="#94a3b8", capsize=3)
        ax.axvline(ref, color="k", ls=":", lw=1)
        ax.set_title(ttl)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
    fig.suptitle(title)
    return _save(fig, path)
