"""Feature-importance data for the dashboard (XGBoost only; deep models use saved permutation importance)."""
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.evaluation.explain import gain_importance, mean_abs_contribution, tree_contributions
from src.models.xgb_models import load_bundle


def xgb_importance(models_dir, ticker: str, feats: pd.DataFrame, recent: int = 250) -> Optional[Dict[str, Optional[pd.Series]]]:
    path = Path(models_dir) / f"{ticker}_xgb_classification.joblib"
    if not path.exists():
        return None
    bundle = load_bundle(path)
    cols, model = bundle["features"], bundle["model"]
    out: Dict[str, Optional[pd.Series]] = {"gain": gain_importance(model, cols), "shap": None}
    X = feats[cols].dropna().tail(recent)
    if len(X):
        try:
            out["shap"] = mean_abs_contribution(tree_contributions(model, X)[0])
        except Exception:  # noqa: BLE001 - SHAP is a bonus; gain importance is still shown
            out["shap"] = None
    return out
