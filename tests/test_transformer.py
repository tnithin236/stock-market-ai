import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.splits import chronological_split  # noqa: E402
from src.features.engineering import build_features, feature_columns, make_supervised  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.models.transformer import PositionalEncoding, TransformerNet  # noqa: E402
from src.training.lstm_fit import fit_task  # noqa: E402
from src.training.nn_trainer import TrainConfig, get_device, predict_array  # noqa: E402
from src.training.sequences import prepare_sequence_data  # noqa: E402


def test_forward_shape():
    out = TransformerNet(6, d_model=16, nhead=2, num_layers=2, dim_feedforward=32)(torch.randn(5, 12, 6))
    assert tuple(out.shape) == (5,)


def test_d_model_must_be_divisible_by_nhead():
    with pytest.raises(ValueError):
        TransformerNet(6, d_model=18, nhead=4)


def test_positional_encoding_has_no_learnable_parameters_and_checks_length():
    pe = PositionalEncoding(16, max_len=20)
    assert len(list(pe.parameters())) == 0
    assert tuple(pe(torch.zeros(2, 20, 16)).shape) == (2, 20, 16)
    with pytest.raises(ValueError):
        pe(torch.zeros(1, 21, 16))


def test_model_is_sensitive_to_the_order_of_days():
    torch.manual_seed(0)
    m = TransformerNet(4, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0).eval()
    x = torch.randn(1, 10, 4)
    with torch.no_grad():
        assert not torch.allclose(m(x), m(x.flip(1)))


def test_factory_builds_both_architectures():
    lstm = build_model("lstm", 5, {"hidden_size": 8, "num_layers": 1, "dropout": 0.1})
    tr = build_model("transformer", 5, {"d_model": 16, "nhead": 2, "num_layers": 1,
                                         "dim_feedforward": 32, "dropout": 0.1})
    x = torch.randn(3, 7, 5)
    assert tuple(lstm(x).shape) == tuple(tr(x).shape) == (3,)
    with pytest.raises(ValueError):
        build_model("gru", 5, {})


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_fit_task_runs_for_the_transformer(ohlcv, task):
    sup = make_supervised(build_features(ohlcv))
    split = chronological_split(sup, 0.7, 0.15, gap=1)
    cols = feature_columns(sup, scale_free_only=True)
    data, _ = prepare_sequence_data(sup, split, cols, 10)
    hp = {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 32, "dropout": 0.1,
          "lr": 0.005, "weight_decay": 1e-4, "batch_size": 32}
    y_std = float(split.train["target_return"].std())
    model, res = fit_task(task, data, hp, TrainConfig(max_epochs=3), get_device(), y_std,
                          0.9 if task == "classification" else None, seed=0, arch="transformer")
    out = predict_array(model, data["test"].X, get_device())
    assert out.shape == (len(data["test"]),) and np.isfinite(out).all()
    assert len(res["history"]["val_loss"]) >= 1
