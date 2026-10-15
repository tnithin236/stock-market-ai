"""Optuna search for sequence models. Uses the VALIDATION split only."""
import logging
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.training.lstm_fit import fit_task
from src.training.nn_trainer import TrainConfig

logger = logging.getLogger(__name__)


def _suggest(trial, arch: str, seq_len_choices: List[int]) -> Dict[str, Any]:
    hp: Dict[str, Any] = {
        "seq_len": trial.suggest_categorical("seq_len", seq_len_choices),
        "num_layers": trial.suggest_int("num_layers", 1, 3 if arch == "transformer" else 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }
    if arch == "lstm":
        hp["hidden_size"] = trial.suggest_categorical("hidden_size", [16, 32, 64])
    else:  # transformer: every d_model below is divisible by every nhead
        hp["d_model"] = trial.suggest_categorical("d_model", [16, 32, 64])
        hp["nhead"] = trial.suggest_categorical("nhead", [2, 4])
        hp["dim_feedforward"] = trial.suggest_categorical("dim_feedforward", [32, 64, 128])
    return hp


def tune_deep(
    arch: str,
    task: str,
    build_fn: Callable[[int], Tuple[Dict[str, Any], Any]],
    base_tc: TrainConfig,
    device,
    y_std: float,
    pos_weight: Optional[float],
    seed: int,
    n_trials: int,
    seq_len_choices: List[int],
    trial_max_epochs: int = 40,
) -> Tuple[Dict[str, Any], float, int]:
    """Minimise the best validation loss. Returns (best_hp, best_value, n_trials_run)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tc = replace(base_tc, max_epochs=min(base_tc.max_epochs, trial_max_epochs))

    def objective(trial) -> float:
        hp = _suggest(trial, arch, seq_len_choices)
        data, _scaler = build_fn(hp["seq_len"])
        _model, res = fit_task(task, data, hp, tc, device, y_std, pos_weight, seed, arch=arch)
        return float(res["best_val_loss"])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    logger.info("%s %s tuning finished: best validation loss %.6f", arch, task, study.best_value)
    return dict(study.best_params), float(study.best_value), len(study.trials)
