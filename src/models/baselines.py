"""Naive baselines every real model must beat.

All baselines read only columns that already exist in the feature frame, which
were computed causally on the FULL history. (Computing a rolling mean on a split
separately would create fake NaNs at the start of validation/test.)
"""
import numpy as np
import pandas as pd


# ---------------- Regression: predict tomorrow's close ----------------
class PreviousDayBaseline:
    name = "previous_day"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["Close"].to_numpy()


class MovingAverageBaseline:
    def __init__(self, window: int):
        self.window = window
        self.name = f"sma_{window}"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        col = f"sma_{self.window}"
        if col not in df.columns:
            raise KeyError(f"{col!r} not in features. Add {self.window} to features.sma_windows.")
        return df[col].to_numpy()


# ---------------- Classification: predict UP (1) or DOWN (0) ----------------
class MajorityClassBaseline:
    """Always predicts the most common training label. The bar for 'accuracy'."""
    name = "always_majority"

    def fit(self, y_train) -> "MajorityClassBaseline":
        y = np.asarray(y_train).astype(int)
        self.prior_up = float(y.mean())
        self.label = int(self.prior_up >= 0.5)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.label, dtype=int)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.prior_up, dtype=float)


class MomentumBaseline:
    """Tomorrow goes the same way as today."""
    name = "momentum (same as today)"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (df["return"] > 0).astype(int).to_numpy()

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return df["return"].to_numpy()


class MeanReversionBaseline:
    """Tomorrow goes the opposite way of today."""
    name = "mean_reversion (opposite)"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (df["return"] < 0).astype(int).to_numpy()

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return -df["return"].to_numpy()
