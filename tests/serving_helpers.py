"""Build a PredictionService wired to synthetic data and tiny trained models (no network)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import chronological_split
from src.features.engineering import build_features, feature_columns, make_supervised
from src.models.calibration import PlattCalibrator
from src.models.xgb_models import fit_model, make_model, predict_proba_up, save_bundle
from src.serving.service import PredictionService
from tests.helpers import make_ohlcv

PARAMS = {"max_depth": 2, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}


def write_results(reports: Path, ticker: str = "AAPL", significant: bool = False) -> None:
    p, ci = (0.001, "[0.560, 0.680]") if significant else (0.73, "[0.493, 0.598]")
    res = {
        "ticker": ticker, "test_period": ["2025-01-17", "2026-10-13"],
        "regression": {"test": [
            {"model": "previous_day", "RMSE": 4.55, "RMSE_vs_naive": 1.0, "oosR2_ret": 0.0, "dir_acc_%": float("nan")},
            {"model": "xgb_reg[scale_free]", "RMSE": 4.56, "RMSE_vs_naive": 1.0019, "oosR2_ret": 0.0003,
             "dir_acc_%": 52.5}]},
        "classification": {"test": [
            {"model": "always_majority", "accuracy": 0.5344, "roc_auc": 0.5, "p_vs_majority": 0.52,
             "acc_CI95": "[0.487, 0.581]", "auc_CI95": "[0.500, 0.500]"},
            {"model": "xgb_raw", "accuracy": 0.5206, "roc_auc": 0.546, "p_vs_majority": p,
             "acc_CI95": "[0.474, 0.567]", "auc_CI95": ci}]},
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"xgboost_results_{ticker}.json").write_text(json.dumps(res))   # contains NaN on purpose


def build_service(tmp: Path, n_rows: int = 900, significant: bool = False, trainer=None) -> PredictionService:
    ohlcv = make_ohlcv(n_rows)
    sup = make_supervised(build_features(ohlcv))
    split = chronological_split(sup, 0.7, 0.15, gap=1)
    cols = feature_columns(sup, scale_free_only=True)

    reg = make_model("regression", PARAMS, n_estimators=30, early_stopping_rounds=5)
    fit_model(reg, split.train[cols], split.train["target_return"], split.val[cols], split.val["target_return"])
    clf = make_model("classification", PARAMS, n_estimators=30, early_stopping_rounds=5)
    fit_model(clf, split.train[cols], split.train["target_direction"], split.val[cols], split.val["target_direction"])
    cal = PlattCalibrator().fit(predict_proba_up(clf, split.val[cols]), split.val["target_direction"])

    models, reports = Path(tmp) / "models", Path(tmp) / "reports"
    models.mkdir(parents=True, exist_ok=True)
    meta = {"feature_set": "scale_free", "trained_on": "2026-10-17", "train_period": ["a", "b"], "val_period": ["c", "d"]}
    save_bundle(models / "AAPL_xgb_regression.joblib", reg, "regression", cols, None, meta)
    save_bundle(models / "AAPL_xgb_classification.joblib", clf, "classification", cols, cal, meta)
    write_results(reports, "AAPL", significant)

    cfg = {"data": {"start_date": "2015-01-01", "interval": "1d", "raw_dir": "data/raw",
                    "cache_max_age_hours": 24, "end_date": None},
           "features": {}, "paths": {"models_dir": str(models), "reports_dir": str(reports)}}
    return PredictionService(cfg, models, reports, data_loader=lambda t: make_ohlcv(n_rows),
                             trainer=trainer or (lambda t, s, e: {"ok": True}))
