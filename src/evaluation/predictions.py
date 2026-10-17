"""Save/load model predictions so the backtester never needs to import model code."""
from pathlib import Path
from typing import List, Union

import pandas as pd

PRED_COLUMNS = ["split", "pred_return", "raw_prob", "cal_prob"]


def save_predictions(path: Union[str, Path], frames: List[pd.DataFrame]) -> None:
    pd.concat(frames).to_csv(path, index_label="Date")


def load_predictions(path: Union[str, Path], split: str = "test") -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    missing = [c for c in PRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; re-run the model command.")
    return df[df["split"] == split].drop(columns=["split"])
