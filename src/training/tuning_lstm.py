"""Optuna search for the LSTM. Uses the VALIDATION split only."""
import logging
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.training.lstm_fit import fit_task
from src.training.nn_trainer import TrainConfig

logger = logging.getLogger(__name__)


def tune_lstm(
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
        hp = {
            "seq_len": trial.suggest_categorical("seq_len", seq_len_choices),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
        data, _scaler = build_fn(hp["seq_len"])
        _model, res = fit_task(task, data, hp, tc, device, y_std, pos_weight, seed)
        return float(res["best_val_loss"])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    logger.info("%s LSTM tuning finished: best validation loss %.6f", task, study.best_value)
    return dict(study.best_params), float(study.best_value), len(study.trials)
