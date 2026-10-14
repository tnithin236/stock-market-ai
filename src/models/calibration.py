"""Probability calibration shared by all classifiers."""
import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Sigmoid (Platt) calibration of raw probabilities.

    Fit on the validation split. With a positive slope it is monotonic, so ROC-AUC
    does not change; it only rescales scores so they behave more like probabilities.

    Safety rule: if the fitted slope is not positive (the raw score looks uninformative
    or anti-correlated with the outcome on validation), we do NOT flip the score's
    direction, because that would just learn noise from ~400 validation labels. We fall
    back to a constant forecast equal to the validation base rate.
    """

    def __init__(self) -> None:
        self.lr = LogisticRegression()
        self.constant = None
        self.slope_ = None

    @staticmethod
    def _logit(p) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw_prob, y) -> "PlattCalibrator":
        y = np.asarray(y).astype(int)
        self.lr.fit(self._logit(raw_prob), y)
        self.slope_ = float(self.lr.coef_[0, 0])
        self.constant = None if self.slope_ > 0 else float(y.mean())
        return self

    def predict(self, raw_prob) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(np.asarray(raw_prob)), self.constant, dtype=float)
        return self.lr.predict_proba(self._logit(raw_prob))[:, 1]