"""Transformer entry point (implementation shared with the LSTM in train_deep.py)."""
from typing import Any, Dict, Optional

from src.training.train_deep import run_deep


def run_transformer(cfg: Dict[str, Any], ticker: Optional[str] = None, start: Optional[str] = None,
                    end: Optional[str] = None, refresh: bool = False, tune: bool = False,
                    n_trials: Optional[int] = None, make_plots: bool = True,
                    save: bool = True) -> Dict[str, Any]:
    return run_deep("transformer", cfg, ticker, start, end, refresh, tune, n_trials, make_plots, save)
