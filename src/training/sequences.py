"""Turn a flat feature table into LSTM/Transformer windows WITHOUT leaking the future.

Sample for target row p (day t):
    X = features of days [t - seq_len + 1, ..., t]      (all known at the close of day t)
    y = what happens on day t + 1                       (target_* columns of row p)

Windows for validation/test samples may reach back into earlier splits. That is
legitimate: it only reads past FEATURES, never labels or future rows. The scaler
is fitted on the training rows only.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.data.splits import DataSplit
from src.features.preprocessing import FeatureScaler


@dataclass
class SequenceSet:
    X: np.ndarray            # (N, seq_len, n_features) float32
    y_ret: np.ndarray        # next-day return
    y_dir: np.ndarray        # next-day direction (0/1)
    index: pd.DatetimeIndex  # date of the LAST day in each window (the prediction date)

    def __len__(self) -> int:
        return len(self.X)


def build_sequences(scaled: np.ndarray, frame: pd.DataFrame, positions, seq_len: int) -> SequenceSet:
    """positions: integer row positions (into `frame`/`scaled`) of the prediction days."""
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    positions = np.asarray(positions, dtype=int)
    positions = positions[positions >= seq_len - 1]   # need a full window of history
    if len(positions) == 0:
        raise ValueError("No samples left after applying seq_len; use a smaller seq_len or more data.")
    windows = sliding_window_view(scaled, seq_len, axis=0)          # (T - L + 1, F, L)
    X = np.ascontiguousarray(windows[positions - seq_len + 1].transpose(0, 2, 1), dtype=np.float32)
    rows = frame.iloc[positions]
    return SequenceSet(
        X=X,
        y_ret=rows["target_return"].to_numpy(dtype=float),
        y_dir=rows["target_direction"].to_numpy(dtype=int),
        index=rows.index,
    )


def prepare_sequence_data(
    sup: pd.DataFrame,
    split: DataSplit,
    columns: List[str],
    seq_len: int,
    scaling_method: str = "standard",
    clip: float = 5.0,
) -> Tuple[Dict[str, SequenceSet], FeatureScaler]:
    """Fit the scaler on TRAIN rows only, scale every row, then window each split."""
    scaler = FeatureScaler(scaling_method).fit(split.train, columns)
    scaled = scaler.transform(sup).to_numpy(dtype=float)
    if clip:
        scaled = np.clip(scaled, -clip, clip)
    out: Dict[str, SequenceSet] = {}
    for name, part in split.items():
        positions = sup.index.get_indexer(part.index)
        out[name] = build_sequences(scaled, sup, positions, seq_len)
    return out, scaler
