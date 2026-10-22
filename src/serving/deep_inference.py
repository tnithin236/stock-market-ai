"""Next-session inference for the saved LSTM / Transformer artifacts (CPU).

Mirrors training exactly: same features, same train-fitted scaler, same +/-clip, same window length.
Artifacts hold only tensors and plain dicts, so torch.load(weights_only=True) is safe.
"""
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.features.preprocessing import FeatureScaler
from src.serving.errors import InsufficientHistoryError, ModelNotTrainedError


def artifact_paths(models_dir, ticker: str, arch: str) -> Dict[str, Path]:
    m = Path(models_dir)
    return {"classification": m / f"{ticker}_{arch}_classification.pt",
            "regression": m / f"{ticker}_{arch}_regression.pt",
            "scaler": m / f"{ticker}_{arch}_scaler.joblib",
            "calibrator": m / f"{ticker}_{arch}_calibrator.joblib"}


def models_exist(models_dir, ticker: str, arch: str) -> bool:
    return all(p.exists() for p in artifact_paths(models_dir, ticker, arch).values())


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


def predict_deep(arch: str, ticker: str, feats: pd.DataFrame, models_dir, clip: float = 5.0) -> Dict[str, Any]:
    """`feats` = build_features(...) output (last row = latest day). Returns the common prediction dict."""
    import torch

    from src.models.factory import build_model

    paths = artifact_paths(models_dir, ticker, arch)
    if not models_exist(models_dir, ticker, arch):
        raise ModelNotTrainedError(
            f"No trained {arch} model for {ticker}. Run `python main.py {arch} --ticker {ticker}` first.")
    scaler = FeatureScaler.load(paths["scaler"])
    calibrator = joblib.load(paths["calibrator"])

    def run(task: str):
        ck = torch.load(paths[task], map_location="cpu", weights_only=True)
        hp, cols = ck["hparams"], ck["features"]
        seq_len = int(hp["seq_len"])
        window = feats.iloc[-seq_len:]
        if len(window) < seq_len or window[scaler.columns].isna().any().any():
            raise InsufficientHistoryError(
                f"Need {seq_len} clean trading days of history for the {arch} model; not enough for {ticker}.")
        scaled = scaler.transform(window).to_numpy(dtype=float)
        if clip:
            scaled = np.clip(scaled, -clip, clip)
        model = build_model(arch, len(cols), hp)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            out = float(model(torch.from_numpy(scaled[None, :, :].astype(np.float32))).item())
        return out, ck

    logit, ck_clf = run("classification")
    reg_out, ck_reg = run("regression")
    score = _sigmoid(logit)
    meta = ck_clf.get("meta", {})
    return {
        "score": score,
        "calibrated": float(calibrator.predict(np.array([score]))[0]),
        "pred_ret": float(reg_out * ck_reg["y_std"]),
        "last_close": float(feats["Close"].iloc[-1]),
        "as_of": feats.index[-1].date(),
        "details": {"model": arch, "trained_on": meta.get("trained_on"), "train_period": meta.get("train_period"),
                    "val_period": meta.get("val_period"), "hyperparameters": ck_clf["hparams"],
                    "prediction_horizon": "next trading session"},
    }
