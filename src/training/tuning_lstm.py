"""Backwards-compatible LSTM entry point; the implementation lives in tuning_deep.py."""
from src.training.tuning_deep import tune_deep


def tune_lstm(task, build_fn, base_tc, device, y_std, pos_weight, seed, n_trials,
              seq_len_choices, trial_max_epochs=40):
    return tune_deep("lstm", task, build_fn, base_tc, device, y_std, pos_weight, seed,
                     n_trials, seq_len_choices, trial_max_epochs)
