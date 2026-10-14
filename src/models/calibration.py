"""Probability calibration shared by all classifiers."""
import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Sigmoid (Platt) calibration of raw probabilities.

    Fit on the validation split. It is monotonic, so ROC-AUC does not change;
    it only rescales scores so they behave more like probabilities.
    """

    def __init__(self) -> None:
        self.lr = LogisticRegression()

    @staticmethod
    def _logit(p) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw_prob, y) -> "PlattCalibrator":
        self.lr.fit(self._logit(raw_prob), np.asarray(y).astype(int))
        return self

    def predict(self, raw_prob) -> np.ndarray:
        return self.lr.predict_proba(self._logit(raw_prob))[:, 1]
