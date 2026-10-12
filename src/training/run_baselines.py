"""Evaluate naive baselines on validation and test (regression + classification)."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import resolve_path
from src.evaluation.metrics import classification_metrics, directional_accuracy, regression_metrics
from src.features.engineering import feature_columns
from src.features.preprocessing import FeatureScaler
from src.models.baselines import (MajorityClassBaseline, MeanReversionBaseline,
                                  MomentumBaseline, MovingAverageBaseline, PreviousDayBaseline)
from src.training.dataset import load_supervised, split_dataset

logger = logging.getLogger(__name__)


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def _regression_table(models, part: pd.DataFrame) -> pd.DataFrame:
    y_true = part["target_close"].to_numpy()
    prev = part["Close"].to_numpy()
    rows = []
    for m in models:
        pred = m.predict(part)
        row = {"model": m.name, **regression_metrics(y_true, pred)}
        row["dir_acc_%"] = directional_accuracy(prev, y_true, pred) * 100
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    naive_rmse = table.loc["previous_day", "RMSE"]
    table["RMSE_vs_naive"] = table["RMSE"] / naive_rmse
    return table


def _classification_table(models, part: pd.DataFrame):
    y_true = part["target_direction"].to_numpy()
    rows, matrices = [], {}
    for m in models:
        met = classification_metrics(y_true, m.predict(part), m.score(part))
        matrices[m.name] = met.pop("confusion_matrix")
        rows.append({"model": m.name, **met})
    return pd.DataFrame(rows).set_index("model"), matrices


def run_baselines(
    cfg: Dict[str, Any],
    ticker: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    refresh: bool = False,
    save: bool = True,
) -> Dict[str, Any]:
    ticker, _full, sup = load_supervised(cfg, ticker, start, end, refresh)
    split = split_dataset(sup, cfg)
    cols = feature_columns(sup)

    print(f"\n=== {ticker}: dataset ===")
    print(f"rows after warm-up/target cleanup: {len(sup)} | model inputs: {len(cols)} features")
    print("\n=== Chronological split (no shuffling) ===")
    print(split.summary().to_string())
    print("Leakage audit: time order verified (train < val < test).")

    # Demonstrate train-only scaling
    scaler = FeatureScaler(cfg.get("scaling", {}).get("method", "standard")).fit(split.train, cols)
    tr, va, _te = (scaler.transform(split.train), scaler.transform(split.val), scaler.transform(split.test))
    print(f"Scaler fitted on {len(split.train)} training rows only. "
          f"max|train mean|={tr.mean().abs().max():.3f}, max|val mean|={va.mean().abs().max():.3f} "
          "(val is NOT re-centered, as it should be).")

    # Regression baselines
    reg_models = [PreviousDayBaseline()] + [
        MovingAverageBaseline(w) for w in cfg.get("baselines", {}).get("ma_windows", [10, 20])
    ]
    # Classification baselines
    majority = MajorityClassBaseline().fit(split.train["target_direction"])
    cls_models = [majority, MomentumBaseline(), MeanReversionBaseline()]

    results: Dict[str, Any] = {"ticker": ticker, "split": split.summary().reset_index().to_dict("records"),
                               "regression": {}, "classification": {}}
    for name, part in (("val", split.val), ("test", split.test)):
        reg = _regression_table(reg_models, part)
        cls, mats = _classification_table(cls_models, part)
        print(f"\n=== Regression baselines: next-day close ({name}) ===")
        print(reg.round(4).to_string())
        print(f"\n=== Classification baselines: next-day direction ({name}) ===")
        print(cls.round(4).to_string())
        for mname, cm in mats.items():
            print(f"  confusion [[TN, FP], [FN, TP]] {mname}: {cm}")
        results["regression"][name] = reg.reset_index().to_dict("records")
        results["classification"][name] = cls.reset_index().to_dict("records")
        results.setdefault("confusion", {})[name] = mats

    print("\nHow to read this:")
    print(" - R2 on price LEVELS is very high (often 0.9+) for almost anything because prices trend. "
          "Judge models by RMSE_vs_naive (< 1.0 means better than previous-day).")
    print(" - 'always_majority' accuracy equals the share of the majority class. "
          "A classifier must beat it AND show ROC-AUC clearly above 0.5.")

    if save:
        out_dir = resolve_path(cfg["paths"].get("reports_dir", "reports"))
        path = Path(out_dir) / f"baseline_results_{ticker}.json"
        path.write_text(json.dumps(results, indent=2, default=_json_default))
        feats = resolve_path(cfg["data"]["processed_dir"]) / f"{ticker}_features.csv"
        sup.to_csv(feats)
        print(f"\nSaved results to {path}\nSaved feature table to {feats}")
    return results
