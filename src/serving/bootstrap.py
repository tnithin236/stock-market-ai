"""Container start-up helper: optionally train a first XGBoost model so a fresh deployment is usable.

    AUTO_TRAIN=1 python -m src.serving.bootstrap          # trains only if the ticker has no model yet

Does nothing unless AUTO_TRAIN=1 (so local runs and tests are never surprised by a download).
"""
import logging
import os
import sys
from pathlib import Path
from typing import Callable

from src.data.loader import normalize_ticker

logger = logging.getLogger(__name__)


def ensure_models(ticker: str, models_dir, train_fn: Callable[[str], None]) -> bool:
    """Train via `train_fn(ticker)` when the XGBoost bundles are missing. Returns True if it trained."""
    t = normalize_ticker(ticker)
    models_dir = Path(models_dir)
    have = [(models_dir / f"{t}_xgb_{task}.joblib").exists() for task in ("classification", "regression")]
    if all(have):
        return False
    train_fn(t)
    return True


def main() -> int:
    if os.environ.get("AUTO_TRAIN", "0") != "1":
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from src.config import load_config, resolve_path
    from src.training.train_xgboost import run_xgboost

    cfg = load_config()
    ticker = os.environ.get("AUTO_TRAIN_TICKER", cfg["data"]["ticker"])
    trained = ensure_models(ticker, resolve_path(cfg["paths"]["models_dir"]),
                            lambda t: run_xgboost(cfg, t, make_plots=False, save=True))
    print(f"bootstrap: {'trained a new model for ' + ticker if trained else 'models already present for ' + ticker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
