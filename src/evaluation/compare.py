"""Side-by-side comparison of every model, built from the saved result files."""
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

RESULT_FILES = ("xgboost", "lstm", "transformer")
_CI = re.compile(r"\[\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\]")


def _parse_ci(text) -> Tuple[float, float]:
    m = _CI.search(str(text))
    return (float(m.group(1)), float(m.group(2))) if m else (float("nan"), float("nan"))


def _rows(results: Dict, task: str, split: str = "test"):
    return results.get(task, {}).get(split, [])


def build_comparison(reports_dir, ticker: str, split: str = "test") -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """Return (regression table, classification table, files found)."""
    reports_dir = Path(reports_dir)
    reg_rows, cls_rows, found = {}, {}, []
    for tag in RESULT_FILES:
        path = reports_dir / f"{tag}_results_{ticker}.json"
        if not path.exists():
            continue
        found.append(path.name)
        res = json.loads(path.read_text())
        for r in _rows(res, "regression", split):
            reg_rows.setdefault(r["model"], r)           # first occurrence wins (shared baselines)
        for r in _rows(res, "classification", split):
            cls_rows.setdefault(r["model"], r)

    reg = pd.DataFrame([{"model": m, "RMSE": r.get("RMSE"), "RMSE_vs_naive": r.get("RMSE_vs_naive"),
                         "oosR2_ret": r.get("oosR2_ret"), "dir_acc_%": r.get("dir_acc_%")}
                        for m, r in reg_rows.items()]).set_index("model") if reg_rows else pd.DataFrame()

    cls_records = []
    for m, r in cls_rows.items():
        lo, hi = _parse_ci(r.get("acc_CI95"))
        alo, ahi = _parse_ci(r.get("auc_CI95"))
        cls_records.append({"model": m, "accuracy": r.get("accuracy"), "acc_lo": lo, "acc_hi": hi,
                            "p_vs_majority": r.get("p_vs_majority"), "roc_auc": r.get("roc_auc"),
                            "auc_lo": alo, "auc_hi": ahi, "brier": r.get("brier")})
    cls = pd.DataFrame(cls_records).set_index("model") if cls_records else pd.DataFrame()
    return reg, cls, found


def significant_models(cls: pd.DataFrame, alpha: float = 0.05) -> list:
    """Models whose accuracy beats the majority baseline (p < alpha) AND whose AUC interval is above 0.5."""
    if cls.empty:
        return []
    ok = (cls["p_vs_majority"] < alpha) & (cls["auc_lo"] > 0.5)
    return list(cls.index[ok.fillna(False)])
