"""Chronological train / validation / test split (never shuffled)."""
from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def items(self):
        return [("train", self.train), ("val", self.val), ("test", self.test)]

    def summary(self, direction_col: str = "target_direction") -> pd.DataFrame:
        rows = []
        for name, part in self.items():
            row: Dict[str, object] = {
                "split": name,
                "rows": len(part),
                "start": str(part.index.min().date()),
                "end": str(part.index.max().date()),
            }
            if direction_col in part.columns:
                row["up_%"] = round(float(part[direction_col].mean() * 100), 2)
            rows.append(row)
        return pd.DataFrame(rows).set_index("split")


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    gap: int = 1,
) -> DataSplit:
    """Split by time order.

    gap: rows removed at the start of validation and test. The label of the last
    training row is tomorrow's price, which is the first validation day, so a gap
    of 1 stops training labels from touching the next set.
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Need 0 < train_frac, val_frac and train_frac + val_frac < 1")
    if gap < 0:
        raise ValueError("gap must be >= 0")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted ascending before splitting.")

    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    split = DataSplit(
        train=df.iloc[:i_train],
        val=df.iloc[i_train + gap:i_val],
        test=df.iloc[i_val + gap:],
    )
    for name, part in split.items():
        if len(part) == 0:
            raise ValueError(f"{name} split is empty; need more data or different fractions.")
    audit_split(split)
    return split


def audit_split(split: DataSplit) -> None:
    """Hard checks that fail loudly if time order is violated."""
    if not split.train.index.max() < split.val.index.min():
        raise AssertionError("Leakage: train overlaps validation in time")
    if not split.val.index.max() < split.test.index.min():
        raise AssertionError("Leakage: validation overlaps test in time")
