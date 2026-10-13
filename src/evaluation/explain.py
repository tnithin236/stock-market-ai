"""Explainability for XGBoost: native TreeSHAP contributions and gain importance."""
from typing import Tuple

import numpy as np
import pandas as pd


def tree_contributions(model, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Exact TreeSHAP values from XGBoost itself (no extra library needed).

    Returns (contributions per feature, bias). For the regressor the unit is
    'return'; for the classifier it is log-odds. contributions.sum(axis=1) + bias
    equals the model's raw margin output.
    """
    import xgboost as xgb

    booster = model.get_booster()
    dmat = xgb.DMatrix(X)
    contrib = booster.predict(dmat, pred_contribs=True,
                              iteration_range=(0, int(model.best_iteration) + 1))
    return pd.DataFrame(contrib[:, :-1], index=X.index, columns=X.columns), contrib[:, -1]


def mean_abs_contribution(contrib: pd.DataFrame) -> pd.Series:
    return contrib.abs().mean().sort_values(ascending=False)


def gain_importance(model, columns) -> pd.Series:
    """Share of total split gain per feature, counting ONLY the trees used for prediction.

    Early stopping leaves extra trees after the best round. predict() ignores them,
    so importance must ignore them too.
    """
    booster = model.get_booster()
    try:
        booster = booster[: int(model.best_iteration) + 1]
    except Exception:
        pass
    raw = booster.get_score(importance_type="gain")
    s = pd.Series({c: float(raw.get(c, 0.0)) for c in columns})
    total = s.sum()
    return (s / total if total > 0 else s).sort_values(ascending=False)