"""Read saved experiment outputs for the dashboard (no Streamlit import)."""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.evaluation.predictions import load_predictions

MODEL_LABELS = {"xgboost": "XGBoost", "lstm": "LSTM", "transformer": "Transformer"}


def load_results(reports_dir, ticker: str, tag: str) -> Optional[Dict[str, Any]]:
    path = Path(reports_dir) / f"{tag}_results_{ticker}.json"
    return json.loads(path.read_text()) if path.exists() else None


def results_tables(results: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(regression, classification) tables for the TEST period."""
    reg = pd.DataFrame(results.get("regression", {}).get("test", []))
    cls = pd.DataFrame(results.get("classification", {}).get("test", []))
    return (reg.set_index("model") if len(reg) else reg), (cls.set_index("model") if len(cls) else cls)


def permutation_table(results: Dict[str, Any], task: str = "classifier") -> Optional[pd.DataFrame]:
    rows = results.get("permutation_importance", {}).get(task)
    return pd.DataFrame(rows).set_index("feature") if rows else None


def prediction_frame(reports_dir, ticker: str, tag: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Test-period predictions aligned with actual next-day closes from `df`."""
    path = Path(reports_dir) / f"predictions_{tag}_{ticker}.csv"
    if not path.exists():
        return None
    p = load_predictions(path, "test")
    close = df["Close"].reindex(p.index)
    frame = pd.DataFrame({
        "close": close,
        "actual_next_close": df["Close"].shift(-1).reindex(p.index),
        "model_pred_close": close * (1.0 + p["pred_return"]),
        "raw_prob": p["raw_prob"],
    }).dropna()
    return frame if len(frame) else None


def direction_summary(frame: pd.DataFrame) -> Dict[str, float]:
    actual_up = frame["actual_next_close"] > frame["close"]
    pred_up = frame["raw_prob"] > 0.5
    return {"n_days": int(len(frame)), "accuracy": float((actual_up == pred_up).mean()),
            "always_up_accuracy": float(actual_up.mean())}
