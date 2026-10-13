"""Hyperparameter search with Optuna. Uses the VALIDATION split only."""
import logging
from typing import Any, Dict, Optional, Tuple

from src.models.xgb_models import fit_model, make_model

logger = logging.getLogger(__name__)


def tune_xgboost(
    task: str,
    X_train, y_train, X_val, y_val,
    n_trials: int = 30,
    seed: int = 42,
    scale_pos_weight: Optional[float] = None,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
) -> Tuple[Dict[str, Any], float, int]:
    """Minimise validation RMSE (regression) or validation log-loss (classification).

    n_estimators is tuned implicitly: it is set high and early stopping picks the
    best round for every trial. Returns (best_params, best_value, n_trials_run).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        }
        model = make_model(task, params, n_estimators, early_stopping_rounds, seed, scale_pos_weight)
        fit_model(model, X_train, y_train, X_val, y_val)
        return float(model.best_score)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    logger.info("%s tuning finished: best validation loss %.6f", task, study.best_value)
    return dict(study.best_params), float(study.best_value), len(study.trials)
