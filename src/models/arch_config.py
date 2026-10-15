"""Per-architecture hyperparameter names and defaults (no torch import, so it is cheap to load)."""
from typing import Any, Dict

ARCH_LABEL = {"lstm": "LSTM", "transformer": "Transformer"}

ARCH_HP_KEYS = {
    "lstm": ("seq_len", "hidden_size", "num_layers", "dropout", "lr", "weight_decay", "batch_size"),
    "transformer": ("seq_len", "d_model", "nhead", "num_layers", "dim_feedforward", "dropout",
                    "lr", "weight_decay", "batch_size"),
}

COMMON_DEFAULTS: Dict[str, Any] = {
    "feature_set": "scale_free", "seq_len": 30, "weight_decay": 1e-4, "batch_size": 64,
    "max_epochs": 60, "patience": 10, "lr_patience": 3, "lr_factor": 0.5, "grad_clip": 1.0,
    "clip_scaled": 5.0, "seed": 42, "use_class_weight": True,
    "tuning": {"n_trials": 12, "seq_len_choices": [10, 20, 30, 60], "max_epochs": 40},
    "permutation": {"n_repeats": 5},
}

ARCH_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "lstm": {"hidden_size": 32, "num_layers": 1, "dropout": 0.2, "lr": 1e-3},
    "transformer": {"d_model": 32, "nhead": 4, "num_layers": 2, "dim_feedforward": 64,
                    "dropout": 0.1, "lr": 5e-4},
}


def merged_deep_config(cfg: Dict[str, Any], arch: str) -> Dict[str, Any]:
    """Defaults < arch defaults < the `<arch>:` section of config.yaml."""
    if arch not in ARCH_DEFAULTS:
        raise ValueError(f"Unknown architecture {arch!r}; choose from {sorted(ARCH_DEFAULTS)}")
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in COMMON_DEFAULTS.items()}
    out.update(ARCH_DEFAULTS[arch])
    for k, v in (cfg.get(arch) or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out
