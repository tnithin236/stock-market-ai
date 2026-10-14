"""Model-agnostic PyTorch training loop (reused by the LSTM and the Transformer)."""
import copy
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 10        # early stopping
    lr_patience: int = 3      # ReduceLROnPlateau
    lr_factor: float = 0.5
    grad_clip: float = 1.0


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    """Shuffling here only reorders TRAINING WINDOWS inside an epoch; each window already
    contains only past data, so this is not a time-series leak."""
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                       torch.from_numpy(np.asarray(y, dtype=np.float32)))
    gen = None
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


def _train_loss(task: str, pos_weight: Optional[float], device: torch.device) -> nn.Module:
    if task == "regression":
        return nn.HuberLoss(delta=1.0)               # robust to fat-tailed returns
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device) if pos_weight else None
    return nn.BCEWithLogitsLoss(pos_weight=pw)


def _eval_loss(task: str) -> nn.Module:
    # Validation uses the UNWEIGHTED loss (a proper scoring rule) for early stopping and tuning.
    return nn.HuberLoss(delta=1.0) if task == "regression" else nn.BCEWithLogitsLoss()


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += loss_fn(model(xb), yb).item() * len(xb)
        n += len(xb)
    return total / max(n, 1)


def train_model(
    model: nn.Module,
    task: str,
    train_xy: Tuple[np.ndarray, np.ndarray],
    val_xy: Tuple[np.ndarray, np.ndarray],
    tc: TrainConfig,
    device: torch.device,
    pos_weight: Optional[float] = None,
    ckpt_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train with AdamW, ReduceLROnPlateau, gradient clipping, early stopping and checkpointing.

    Early stopping and scheduling watch the VALIDATION loss. The best weights are
    restored at the end. The test set is never touched here.
    """
    model.to(device)
    train_loader = make_loader(train_xy[0], train_xy[1], tc.batch_size, True, seed)
    val_loader = make_loader(val_xy[0], val_xy[1], 512, False)
    train_fn, val_fn = _train_loss(task, pos_weight, device), _eval_loss(task)

    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=tc.lr_factor, patience=tc.lr_patience)

    best_val, best_epoch, best_state, bad = float("inf"), 0, None, 0
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}
    epoch = 0
    for epoch in range(1, tc.max_epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = train_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            running += loss.item() * len(xb)
            seen += len(xb)
        train_loss = running / max(seen, 1)
        val_loss = _evaluate(model, val_loader, val_fn, device)
        sched.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(opt.param_groups[0]["lr"])

        if val_loss < best_val - 1e-6:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            if ckpt_path is not None:
                torch.save({"epoch": epoch, "val_loss": val_loss, "state_dict": best_state}, ckpt_path)
        else:
            bad += 1
            if bad >= tc.patience:
                break

    if best_state is None:
        raise RuntimeError("Training never produced a finite validation loss.")
    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": float(best_val),
            "epochs_run": epoch}


@torch.no_grad()
def predict_array(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """Raw model outputs (scaled returns or logits), computed in eval mode."""
    model.eval()
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    outs = [model(Xt[i:i + batch_size].to(device)).cpu().numpy() for i in range(0, len(Xt), batch_size)]
    return np.concatenate(outs) if outs else np.empty(0, dtype=np.float32)
