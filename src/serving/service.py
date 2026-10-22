"""Framework-free serving logic (so it can be unit-tested without FastAPI).

Predictions come from the saved XGBoost bundles. The service is deliberately explicit that the
models have shown no statistically significant predictive edge, and it never presents a raw score
as a probability.
"""
import copy
import json
import logging
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.loader import load_stock_data, normalize_ticker
from src.features.engineering import build_features
from src.models.xgb_models import load_bundle, predict_with_bundle
from src.models.arch_config import merged_deep_config
from src.serving import deep_inference
from src.serving.errors import (DataUnavailableError, InsufficientHistoryError, ModelNotTrainedError)
from src.serving.jobs import JobRegistry

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"
DISCLAIMER = "Experimental research/educational system - NOT financial advice. Do not trade on this output."
MAX_STALENESS_DAYS = 6
MODEL_NAMES = ("xgboost", "lstm", "transformer")


def clean(obj: Any) -> Any:
    """Make results JSON-safe: NaN/inf -> None, numpy -> python, dates -> str (FastAPI rejects NaN)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return str(obj)
    return obj


class PredictionService:
    def __init__(
        self,
        cfg: Dict[str, Any],
        models_dir: Path,
        reports_dir: Path,
        data_loader: Optional[Callable[[str], pd.DataFrame]] = None,
        trainer: Optional[Callable[[str, Optional[str], Optional[str]], Dict[str, Any]]] = None,
    ) -> None:
        self.cfg = cfg
        self.models_dir, self.reports_dir = Path(models_dir), Path(reports_dir)
        self.data_loader = data_loader or self._default_loader
        self.trainer = trainer or self._default_trainer
        self.jobs = JobRegistry()
        self._bundles: Dict[str, Any] = {}

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_config(cls, config_path: str = "configs/config.yaml") -> "PredictionService":
        cfg = load_config(config_path)
        return cls(cfg, resolve_path(cfg["paths"]["models_dir"]),
                   resolve_path(cfg["paths"].get("reports_dir", "reports")))

    def _default_loader(self, ticker: str) -> pd.DataFrame:
        d = self.cfg["data"]
        return load_stock_data(ticker, d["start_date"], None, d["interval"], d["raw_dir"],
                               d["cache_max_age_hours"], force_refresh=False)

    def _default_trainer(self, ticker: str, start: Optional[str], end: Optional[str]) -> Dict[str, Any]:
        from src.training.train_xgboost import run_xgboost

        cfg = copy.deepcopy(self.cfg)
        cfg["data"]["end_date"] = end          # API training uses live data unless an end date is given
        res = run_xgboost(cfg, ticker, start, end, refresh=False, tune=False, make_plots=False, save=True)
        self._bundles.pop(ticker, None)
        return {"test_period": res.get("test_period"), "chosen_features": res.get("chosen_features")}

    # ---------------------------------------------------------------- helpers
    def _load(self, ticker: str) -> pd.DataFrame:
        try:
            df = self.data_loader(ticker)
        except Exception as exc:  # noqa: BLE001
            raise DataUnavailableError(f"Could not load market data for {ticker}: {exc}") from exc
        if df is None or len(df) == 0:
            raise DataUnavailableError(f"No market data returned for {ticker}.")
        return df

    def _bundle_paths(self, ticker: str):
        return (self.models_dir / f"{ticker}_xgb_classification.joblib",
                self.models_dir / f"{ticker}_xgb_regression.joblib")

    def _get_bundles(self, ticker: str):
        clf_path, reg_path = self._bundle_paths(ticker)
        if not (clf_path.exists() and reg_path.exists()):
            raise ModelNotTrainedError(
                f"No trained model for {ticker}. Train one first with POST /train (or `python main.py xgboost`).")
        stamp = (clf_path.stat().st_mtime, reg_path.stat().st_mtime)
        cached = self._bundles.get(ticker)
        if cached is None or cached[0] != stamp:
            cached = (stamp, load_bundle(clf_path), load_bundle(reg_path))
            self._bundles[ticker] = cached
        return cached[1], cached[2]

    def trained_tickers(self) -> List[str]:
        if not self.models_dir.exists():
            return []
        out = []
        for p in sorted(self.models_dir.glob("*_xgb_classification.joblib")):
            t = p.name[: -len("_xgb_classification.joblib")]
            if (self.models_dir / f"{t}_xgb_regression.joblib").exists():
                out.append(t)
        return out

    def _read_results(self, ticker: str, model: str = "xgboost") -> Optional[Dict[str, Any]]:
        path = self.reports_dir / f"{model}_results_{ticker}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def _evidence(self, ticker: str, model: str = "xgboost") -> Optional[Dict[str, Any]]:
        res = self._read_results(ticker, model)
        if not res:
            return None
        rows = {r["model"]: r for r in res.get("classification", {}).get("test", [])}
        raw, base = rows.get("xgb_raw" if model == "xgboost" else f"{model}_raw"), rows.get("always_majority")
        if not raw or not base:
            return None
        auc_ci = str(raw.get("auc_CI95", ""))
        try:
            auc_lo = float(auc_ci.strip("[]").split(",")[0])
        except ValueError:
            auc_lo = float("nan")
        p = raw.get("p_vs_majority")
        significant = bool(p is not None and p < 0.05 and auc_lo > 0.5)
        return clean({
            "test_period": res.get("test_period"), "test_accuracy": raw.get("accuracy"),
            "always_up_accuracy": base.get("accuracy"), "roc_auc": raw.get("roc_auc"), "roc_auc_95ci": auc_ci,
            "p_vs_majority": p, "significant_edge": significant,
            "note": "Measured on a held-out test period; wide intervals mean small differences are noise.",
        })

    def evidence(self, ticker: str, model: str = "xgboost") -> Optional[Dict[str, Any]]:
        """Held-out test evidence for a model (public wrapper used by the dashboard)."""
        return self._evidence(normalize_ticker(ticker), model)

    def available_models(self, ticker: str) -> List[str]:
        out = []
        clf_path, reg_path = self._bundle_paths(ticker)
        if clf_path.exists() and reg_path.exists():
            out.append("xgboost")
        out += [a for a in ("lstm", "transformer") if deep_inference.models_exist(self.models_dir, ticker, a)]
        return out

    # ---------------------------------------------------------------- public operations
    def health(self) -> Dict[str, Any]:
        tickers = self.trained_tickers()
        return {"status": "ok", "version": API_VERSION, "trained_tickers": tickers,
                "models": {t: self.available_models(t) for t in tickers},
                "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def get_stock(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        t = normalize_ticker(ticker)
        df = self._load(t).tail(int(days))
        rows = [{"date": str(i.date()), "open": float(r.Open), "high": float(r.High), "low": float(r.Low),
                 "close": float(r.Close), "volume": float(r.Volume)} for i, r in df.iterrows()]
        rets = df["Close"].pct_change().dropna()
        vol = float(rets.std() * np.sqrt(252) * 100) if len(rets) > 1 else None
        return clean({
            "ticker": t, "rows": rows, "latest_close": rows[-1]["close"],
            "period_return_pct": float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100),
            "annualized_volatility_pct": vol,
        })

    def _xgb_outputs(self, t: str, feats: pd.DataFrame) -> Dict[str, Any]:
        clf, reg = self._get_bundles(t)
        last = feats.iloc[[-1]]
        needed = sorted(set(clf["features"]) | set(reg["features"]))
        if last[needed].isna().any().any():
            raise InsufficientHistoryError(
                f"Not enough history for {t} to compute every indicator (need roughly 60+ trading days).")
        out_c, out_r = predict_with_bundle(clf, last), predict_with_bundle(reg, last)
        meta = clf.get("meta", {})
        return {
            "score": float(out_c["raw_score"][0]),
            "calibrated": float(out_c["calibrated_prob"][0]) if "calibrated_prob" in out_c else None,
            "pred_ret": float(out_r["pred_return"][0]),
            "last_close": float(last["Close"].iloc[0]),
            "as_of": last.index[0].date(),
            "details": {"model": "xgboost", "feature_set": meta.get("feature_set"), "trained_on": meta.get("trained_on"),
                        "train_period": meta.get("train_period"), "val_period": meta.get("val_period"),
                        "prediction_horizon": "next trading session"},
        }

    def predict(self, ticker: str, model: str = "xgboost") -> Dict[str, Any]:
        if model not in MODEL_NAMES:
            raise ValueError(f"model must be one of {MODEL_NAMES}")
        t = normalize_ticker(ticker)
        if model == "xgboost":
            self._get_bundles(t)                                   # 404 before touching the network
        elif not deep_inference.models_exist(self.models_dir, t, model):
            raise ModelNotTrainedError(
                f"No trained {model} model for {t}. Run `python main.py {model} --ticker {t}` first.")
        df = self._load(t)
        feats = build_features(df, self.cfg.get("features"))
        if model == "xgboost":
            o = self._xgb_outputs(t, feats)
        else:
            clip = merged_deep_config(self.cfg, model)["clip_scaled"]
            o = deep_inference.predict_deep(model, t, feats, self.models_dir, clip)

        evidence = self._evidence(t, model)
        warnings = [DISCLAIMER]
        if evidence is None or not evidence.get("significant_edge"):
            warnings.append("No statistically significant predictive edge has been demonstrated for this model.")
        if (date.today() - o["as_of"]).days > MAX_STALENESS_DAYS:
            warnings.append(f"Latest available bar is {o['as_of']}; data may be stale.")
        return clean({
            "ticker": t,
            "as_of_date": str(o["as_of"]),
            "prediction": "UP" if o["score"] > 0.5 else "DOWN",
            "predicted_close": o["last_close"] * (1.0 + o["pred_ret"]),
            "predicted_return_pct": o["pred_ret"] * 100,
            "last_close": o["last_close"],
            "score": o["score"],
            "score_is_probability": False,
            "calibrated_probability_up": o["calibrated"],
            "calibration_note": ("Platt-calibrated on a validation period. For a weak signal it stays close to the "
                                 "historical share of up days, so do not read it as confidence in this prediction."),
            "evidence": evidence,
            "details": o["details"],
            "warnings": warnings,
            "disclaimer": DISCLAIMER,
        })

    def metrics(self, ticker: str, model: str = "xgboost") -> Dict[str, Any]:
        if model not in MODEL_NAMES:
            raise ValueError(f"model must be one of {MODEL_NAMES}")
        t = normalize_ticker(ticker)
        res = self._read_results(t, model)
        if res is None:
            raise ModelNotTrainedError(f"No saved evaluation results for {t}. Run training first.")
        backtest = None
        bt_path = self.reports_dir / f"backtest_summary_{t}.csv"
        if bt_path.exists():
            bt = pd.read_csv(bt_path, index_col=0)
            keep = [n for n in ("buy_and_hold", f"{model}_clf", f"{model}_reg") if n in bt.index]
            backtest = {n: bt.loc[n].to_dict() for n in keep}
        evidence = self._evidence(t, model)
        text = ("No statistically significant predictive edge was found on the held-out test period."
                if not (evidence and evidence.get("significant_edge")) else
                "The model beat the majority-class baseline with statistical significance on the test period; "
                "treat with caution and verify out of sample.")
        return clean({
            "ticker": t, "test_period": res.get("test_period"),
            "regression": res.get("regression", {}).get("test", []),
            "classification": res.get("classification", {}).get("test", []),
            "backtest": backtest, "interpretation": text, "disclaimer": DISCLAIMER,
        })

    def start_training(self, ticker: str, start: Optional[str], end: Optional[str]):
        """Validate input, register a job and return (job, callable to run in the background)."""
        t = normalize_ticker(ticker)
        for label, value in (("start_date", start), ("end_date", end)):
            if value is not None:
                try:
                    pd.Timestamp(value)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{label} must be a valid date like 2015-01-01") from exc
        job = self.jobs.create(t)
        start = start or self.cfg["data"]["start_date"]
        return job, (lambda: self.jobs.run(job["job_id"], lambda: self.trainer(t, start, end)))
