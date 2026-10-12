"""Glue: raw data -> features -> supervised table -> chronological splits."""
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.data.loader import load_stock_data, normalize_ticker
from src.data.splits import DataSplit, chronological_split
from src.features.engineering import build_features, make_supervised


def load_supervised(
    cfg: Dict[str, Any],
    ticker: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    refresh: bool = False,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """Return (ticker, full_feature_frame, supervised_frame)."""
    d = cfg["data"]
    ticker = normalize_ticker(ticker or d["ticker"])
    raw = load_stock_data(
        ticker, start or d["start_date"], end or d["end_date"], d["interval"],
        d["raw_dir"], d["cache_max_age_hours"], force_refresh=refresh,
    )
    full = build_features(raw, cfg.get("features"))
    return ticker, full, make_supervised(full)


def split_dataset(supervised: pd.DataFrame, cfg: Dict[str, Any]) -> DataSplit:
    s = cfg.get("split", {})
    return chronological_split(
        supervised, s.get("train_frac", 0.70), s.get("val_frac", 0.15), s.get("gap", 1)
    )
