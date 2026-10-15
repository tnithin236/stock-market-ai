"""Fit one sequence model for one task (shared by the final run and by hyperparameter tuning)."""
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.models.factory import build_model
from src.training.nn_trainer import TrainConfig, set_seed, train_model
from src.training.sequences import SequenceSet


def task_targets(seq: SequenceSet, task: str, y_std: float) -> np.ndarray:
    """Regression target = next-day return divided by the TRAIN std (keeps the loss O(1)).
    Classification target = 0/1 direction."""
    if task == "regression":
        return (seq.y_ret / y_std).astype(np.float32)
    return seq.y_dir.astype(np.float32)


def fit_task(
    task: str,
    data: Dict[str, SequenceSet],
    hp: Dict[str, Any],
    base_tc: TrainConfig,
    device,
    y_std: float,
    pos_weight: Optional[float],
    seed: int,
    ckpt_path=None,
    arch: str = "lstm",
) -> Tuple[Any, Dict[str, Any]]:
    set_seed(seed)
    tc = replace(base_tc, lr=hp["lr"], weight_decay=hp["weight_decay"], batch_size=hp["batch_size"])
    model = build_model(arch, data["train"].X.shape[2], hp)
    result = train_model(
        model, task,
        (data["train"].X, task_targets(data["train"], task, y_std)),
        (data["val"].X, task_targets(data["val"], task, y_std)),
        tc, device,
        pos_weight=pos_weight if task == "classification" else None,
        ckpt_path=ckpt_path, seed=seed,
    )
    return model, result
