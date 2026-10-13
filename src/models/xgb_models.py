"""XGBoost wrappers: construction, early-stopped fitting, prediction, calibration, persistence."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

DEFAULT_XGB_CONFIG: Dict[str, Any] = {
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "seed": 42,
    "use_class_weight": True,
    "params": {
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
    },
    "tuning": {"n_trials": 30},
}

TARGETS = {"regression": "target_return", "classification": "target_direction"}


def merged_xgb_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_XGB_CONFIG.items()}
    for k, v in (cfg.get("xgboost") or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def make_model(
    task: str,
    params: Dict[str, Any],
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
    seed: int = 42,
    scale_pos_weight: Optional[float] = None,
):
    common = dict(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
        tree_method="hist",
        **params,
    )
    if task == "regression":
        return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
    if task == "classification":
        if scale_pos_weight is not None:
            common["scale_pos_weight"] = scale_pos_weight
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    raise ValueError("task must be 'regression' or 'classification'")


def fit_model(model, X_train, y_train, X_val, y_val):
    """Fit with early stopping on the VALIDATION set (never the test set)."""
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _best_range(model):
    return (0, int(model.best_iteration) + 1)


def predict_return(model, X) -> np.ndarray:
    return np.asarray(model.predict(X, iteration_range=_best_range(model)), dtype=float)


def predict_proba_up(model, X) -> np.ndarray:
    return np.asarray(model.predict_proba(X, iteration_range=_best_range(model))[:, 1], dtype=float)


def return_to_price(close, pred_return) -> np.ndarray:
    return np.asarray(close, dtype=float) * (1.0 + np.asarray(pred_return, dtype=float))


class PlattCalibrator:
    """Sigmoid (Platt) calibration of raw XGBoost probabilities.

    Fit on the validation split. It is monotonic, so ROC-AUC does not change;
    it only rescales scores so they behave more like probabilities.
    """

    def __init__(self) -> None:
        self.lr = LogisticRegression()

    @staticmethod
    def _logit(p) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw_prob, y) -> "PlattCalibrator":
        self.lr.fit(self._logit(raw_prob), np.asarray(y).astype(int))
        return self

    def predict(self, raw_prob) -> np.ndarray:
        return self.lr.predict_proba(self._logit(raw_prob))[:, 1]


def save_bundle(path: Union[str, Path], model, task: str, features: List[str],
                calibrator: Optional[PlattCalibrator], meta: Dict[str, Any]) -> None:
    """Persist everything needed for inference. Load only files you created yourself (pickle)."""
    joblib.dump({"model": model, "task": task, "features": list(features),
                 "calibrator": calibrator, "meta": meta}, path)


def load_bundle(path: Union[str, Path]) -> Dict[str, Any]:
    return joblib.load(path)


def predict_with_bundle(bundle: Dict[str, Any], X: pd.DataFrame) -> Dict[str, np.ndarray]:
    X = X[bundle["features"]]
    if bundle["task"] == "regression":
        return {"pred_return": predict_return(bundle["model"], X)}
    raw = predict_proba_up(bundle["model"], X)
    out = {"raw_score": raw}
    if bundle["calibrator"] is not None:
        out["calibrated_prob"] = bundle["calibrator"].predict(raw)
    return out
