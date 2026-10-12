"""Scaling that is fitted on the training split ONLY."""
from pathlib import Path
from typing import List, Optional, Union

import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.data.splits import DataSplit


class FeatureScaler:
    """Thin wrapper around sklearn scalers that keeps DataFrame structure.

    Leakage rule: fit() sees training rows only. Validation/test/live data are
    transformed with the training mean/std, exactly like production would.
    """

    def __init__(self, method: str = "standard"):
        if method not in ("standard", "robust"):
            raise ValueError("method must be 'standard' or 'robust'")
        self.method = method
        self.columns: Optional[List[str]] = None
        self.scaler = StandardScaler() if method == "standard" else RobustScaler()

    def fit(self, train_df: pd.DataFrame, columns: List[str]) -> "FeatureScaler":
        self.columns = list(columns)
        self.scaler.fit(train_df[self.columns])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            raise RuntimeError("Scaler is not fitted. Call fit() on the training split first.")
        scaled = self.scaler.transform(df[self.columns])
        return pd.DataFrame(scaled, index=df.index, columns=self.columns)

    def fit_transform_split(self, split: DataSplit, columns: List[str]):
        self.fit(split.train, columns)
        return (self.transform(split.train), self.transform(split.val), self.transform(split.test))

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump({"method": self.method, "columns": self.columns, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureScaler":
        blob = joblib.load(path)
        obj = cls(blob["method"])
        obj.columns, obj.scaler = blob["columns"], blob["scaler"]
        return obj
