"""Build a sequence model from its architecture name and hyperparameters."""
from typing import Any, Dict


def build_model(arch: str, n_features: int, hp: Dict[str, Any]):
    if arch == "lstm":
        from src.models.lstm import LSTMNet
        return LSTMNet(n_features, hp["hidden_size"], hp["num_layers"], hp["dropout"])
    if arch == "transformer":
        from src.models.transformer import TransformerNet
        return TransformerNet(n_features, hp["d_model"], hp["nhead"], hp["num_layers"],
                              hp["dim_feedforward"], hp["dropout"])
    raise ValueError(f"Unknown architecture {arch!r}")
