"""Small statistical helpers so we do not over-read noisy results."""
from typing import Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion (e.g. accuracy = k/n)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def accuracy_vs_rate_pvalue(k: int, n: int, baseline_rate: float) -> float:
    """One-sided binomial test: is accuracy k/n really above the baseline rate?"""
    baseline_rate = min(max(baseline_rate, 1e-9), 1 - 1e-9)
    return float(stats.binomtest(int(k), int(n), baseline_rate, alternative="greater").pvalue)


def bootstrap_auc_ci(y_true, y_score, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap CI for ROC-AUC (i.i.d. resampling; slightly optimistic for time series)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        aucs.append(roc_auc_score(yt, y_score[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def r2_vs_zero(y_true, y_pred) -> float:
    """Out-of-sample R2 against a 'no change' forecast: 1 - SSE / sum(y^2).

    > 0 means the model beats predicting a zero return; <= 0 means it does not.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(y_true ** 2))
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
