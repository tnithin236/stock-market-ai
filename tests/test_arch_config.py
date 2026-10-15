import pytest

from src.models.arch_config import ARCH_HP_KEYS, merged_deep_config


def test_defaults_and_overrides():
    cfg = {"transformer": {"d_model": 16, "tuning": {"n_trials": 3}}}
    lc = merged_deep_config(cfg, "transformer")
    assert lc["d_model"] == 16 and lc["nhead"] == 4          # override + default
    assert lc["tuning"]["n_trials"] == 3 and lc["tuning"]["max_epochs"] == 40   # nested merge
    assert all(k in lc for k in ARCH_HP_KEYS["transformer"])


def test_lstm_and_transformer_have_different_hyperparameters():
    assert "hidden_size" in ARCH_HP_KEYS["lstm"] and "hidden_size" not in ARCH_HP_KEYS["transformer"]
    assert "d_model" in ARCH_HP_KEYS["transformer"]


def test_unknown_architecture_raises():
    with pytest.raises(ValueError):
        merged_deep_config({}, "gru")
