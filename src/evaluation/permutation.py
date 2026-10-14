"""Permutation importance for sequence models.

For each feature, shuffle its values across samples (keeping each window's
internal time order) and measure how much the score drops.

Limitations (say these out loud in an interview):
  * Correlated features share credit; permuting one creates unrealistic inputs.
  * It measures reliance on a feature, not causal importance.
  * With a near-random model, importances are ~0 +/- noise. Compare to the std.
"""
from typing import Callable, List, Optional

import numpy as np
import pandas as pd


def permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_repeats: int = 5,
    seed: int = 42,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """metric_fn(y, scores) must be 'higher is better'. Importance = baseline - permuted metric."""
    rng = np.random.default_rng(seed)
    n_features = X.shape[2]
    names = feature_names or [f"f{i}" for i in range(n_features)]
    base = metric_fn(y, predict_fn(X))
    rows = []
    for f in range(n_features):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, :, f] = X[perm, :, f]
            drops.append(base - metric_fn(y, predict_fn(Xp)))
        rows.append({"feature": names[f], "importance_mean": float(np.mean(drops)),
                     "importance_std": float(np.std(drops))})
    return pd.DataFrame(rows).set_index("feature").sort_values("importance_mean", ascending=False)
