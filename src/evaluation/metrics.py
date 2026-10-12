"""Regression and classification metrics."""
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, r2_score,
                             recall_score, roc_auc_score)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    nonzero = y_true != 0
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else float("nan"),
        "R2": float(r2_score(y_true, y_pred)),
    }


def directional_accuracy(prev_close, y_true, y_pred) -> float:
    """Share of days where the predicted price move has the right sign.

    Returns NaN when the prediction equals the previous close everywhere
    (a zero-information forecast such as the previous-day baseline).
    """
    prev_close = np.asarray(prev_close, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.all(y_pred == prev_close):
        return float("nan")
    return float(np.mean((y_pred > prev_close) == (y_true > prev_close)))


def classification_metrics(y_true, y_pred, y_score: Optional[np.ndarray] = None) -> Dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    auc = None
    if y_score is not None and len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, np.asarray(y_score, dtype=float)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        # rows = actual [down, up], columns = predicted [down, up]
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
