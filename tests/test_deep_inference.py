import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.splits import chronological_split  # noqa: E402
from src.data.synthetic import make_ohlcv  # noqa: E402
from src.features.engineering import build_features, feature_columns, make_supervised  # noqa: E402
from src.models.calibration import PlattCalibrator  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.serving.deep_inference import artifact_paths, models_exist, predict_deep  # noqa: E402
from src.serving.errors import InsufficientHistoryError, ModelNotTrainedError  # noqa: E402
from src.training.nn_trainer import predict_array  # noqa: E402
from src.training.sequences import prepare_sequence_data  # noqa: E402

HP = {"lstm": {"seq_len": 12, "hidden_size": 8, "num_layers": 1, "dropout": 0.1},
      "transformer": {"seq_len": 12, "d_model": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 32, "dropout": 0.1}}


def _artifacts(d: Path, arch: str, full, sup, split):
    cols = feature_columns(sup, scale_free_only=True)
    data, scaler = prepare_sequence_data(sup, split, cols, HP[arch]["seq_len"], "standard", 5.0)
    paths = artifact_paths(d, "AAPL", arch)
    torch.manual_seed(0)
    models = {}
    for task in ("classification", "regression"):
        m = build_model(arch, len(cols), HP[arch]).eval()
        models[task] = m
        torch.save({"task": task, "architecture": arch, "state_dict": m.state_dict(), "hparams": HP[arch],
                    "features": cols, "y_std": 0.02, "meta": {"trained_on": "2026-10-18"}}, paths[task])
    scaler.save(paths["scaler"])
    joblib.dump(PlattCalibrator().fit(np.linspace(0.3, 0.7, 200), np.tile([0, 1], 100)), paths["calibrator"])
    return data, models


@pytest.mark.parametrize("arch", ["lstm", "transformer"])
def test_inference_matches_the_training_pipeline_exactly(arch):
    full = build_features(make_ohlcv(900))
    sup = make_supervised(full)
    split = chronological_split(sup, 0.7, 0.15, gap=1)
    with tempfile.TemporaryDirectory() as t:
        data, models = _artifacts(Path(t), arch, full, sup, split)
        assert models_exist(t, "AAPL", arch)
        # features available at the last supervised day, exactly as at training time
        out = predict_deep(arch, "AAPL", full.loc[: sup.index[-1]], t)
    last_x = data["test"].X[-1:]
    expect_logit = float(predict_array(models["classification"], last_x, torch.device("cpu"))[0])
    expect_ret = float(predict_array(models["regression"], last_x, torch.device("cpu"))[0]) * 0.02
    assert np.isclose(out["score"], 1 / (1 + np.exp(-expect_logit)), atol=1e-5)
    assert np.isclose(out["pred_ret"], expect_ret, atol=1e-6)
    assert 0 <= out["calibrated"] <= 1 and out["details"]["model"] == arch


def test_missing_artifacts_and_short_history():
    full = build_features(make_ohlcv(900))
    sup = make_supervised(full)
    split = chronological_split(sup, 0.7, 0.15, gap=1)
    with tempfile.TemporaryDirectory() as t:
        with pytest.raises(ModelNotTrainedError):
            predict_deep("lstm", "AAPL", full, t)
        _artifacts(Path(t), "lstm", full, sup, split)
        with pytest.raises(InsufficientHistoryError):
            predict_deep("lstm", "AAPL", full.iloc[-5:], t)
