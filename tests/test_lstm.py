import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.splits import chronological_split  # noqa: E402
from src.features.engineering import build_features, feature_columns, make_supervised  # noqa: E402
from src.models.lstm import LSTMNet  # noqa: E402
from src.training.lstm_fit import fit_task  # noqa: E402
from src.training.nn_trainer import TrainConfig, get_device, predict_array, set_seed, train_model  # noqa: E402
from src.training.sequences import prepare_sequence_data  # noqa: E402


def _toy(n=300, L=8, F=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, L, F)).astype(np.float32)
    y = (X[:, -1, 0] > 0).astype(np.float32)          # learnable: sign of the last day's feature 0
    return X, y


def test_forward_shape():
    out = LSTMNet(5, 8, 2, 0.1)(torch.randn(4, 10, 5))
    assert tuple(out.shape) == (4,)


def test_training_learns_a_simple_pattern_checkpoints_and_restores_best():
    X, y = _toy()
    Xv, yv = _toy(150, seed=1)
    dev = get_device()
    set_seed(0)
    model = LSTMNet(3, 16, 1, 0.0)
    tc = TrainConfig(lr=0.01, max_epochs=40, patience=8, batch_size=32)
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "best.pt"
        res = train_model(model, "classification", (X, y), (Xv, yv), tc, dev, ckpt_path=ckpt, seed=0)
        assert ckpt.exists()
    probs = 1 / (1 + np.exp(-predict_array(model, Xv, dev)))
    assert ((probs > 0.5) == yv).mean() > 0.75
    h = res["history"]
    assert len(h["train_loss"]) == len(h["val_loss"]) == len(h["lr"]) == res["epochs_run"]
    assert np.isclose(res["best_val_loss"], min(h["val_loss"]))


def test_early_stopping_triggers_on_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6, 3)).astype(np.float32)
    y = rng.integers(0, 2, 200).astype(np.float32)           # pure noise labels
    Xv = rng.normal(size=(100, 6, 3)).astype(np.float32)
    yv = rng.integers(0, 2, 100).astype(np.float32)
    set_seed(0)
    res = train_model(LSTMNet(3, 16, 1, 0.0), "classification", (X, y), (Xv, yv),
                      TrainConfig(lr=0.01, max_epochs=80, patience=3, batch_size=32), get_device(), seed=0)
    assert res["epochs_run"] < 80


def test_prediction_does_not_depend_on_what_else_is_in_the_batch():
    X, _ = _toy(40)
    set_seed(0)
    model = LSTMNet(3, 8, 1, 0.3)
    dev = get_device()
    assert np.allclose(predict_array(model, X, dev)[:5], predict_array(model, X[:5], dev), atol=1e-5)


def test_same_seed_gives_same_model():
    dev = get_device()
    if dev.type != "cpu":
        pytest.skip("bitwise reproducibility is only checked on CPU")
    X, y = _toy(120)
    Xv, yv = _toy(60, seed=2)
    outs = []
    for _ in range(2):
        set_seed(7)
        m = LSTMNet(3, 8, 1, 0.2)
        train_model(m, "classification", (X, y), (Xv, yv), TrainConfig(max_epochs=3, batch_size=32), dev, seed=7)
        outs.append(predict_array(m, Xv, dev))
    assert np.allclose(outs[0], outs[1], atol=1e-6)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_fit_task_runs_on_pipeline_data(ohlcv, task):
    sup = make_supervised(build_features(ohlcv))
    split = chronological_split(sup, 0.7, 0.15, gap=1)
    cols = feature_columns(sup, scale_free_only=True)
    data, _ = prepare_sequence_data(sup, split, cols, 10)
    hp = {"hidden_size": 8, "num_layers": 1, "dropout": 0.1, "lr": 0.01, "weight_decay": 1e-4, "batch_size": 32}
    y_std = float(split.train["target_return"].std())
    model, res = fit_task(task, data, hp, TrainConfig(max_epochs=3), get_device(), y_std,
                          0.9 if task == "classification" else None, seed=0)
    out = predict_array(model, data["test"].X, get_device())
    assert out.shape == (len(data["test"]),) and np.isfinite(out).all()
    assert len(res["history"]["val_loss"]) >= 1
