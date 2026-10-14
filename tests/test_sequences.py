import numpy as np

from src.data.splits import chronological_split
from src.features.engineering import build_features, feature_columns, make_supervised
from src.training.sequences import build_sequences, prepare_sequence_data


def _sup(ohlcv):
    return make_supervised(build_features(ohlcv))


def test_windows_are_consecutive_and_end_on_the_prediction_day(ohlcv):
    T, F, L = 60, 3, 5
    sup = _sup(ohlcv).iloc[:T]
    scaled = np.repeat(np.arange(T, dtype=float)[:, None], F, axis=1)   # row i holds the value i
    seq = build_sequences(scaled, sup, np.arange(10, 40), seq_len=L)
    assert seq.X.shape == (30, L, F)
    for i, p in enumerate(range(10, 40)):
        assert np.array_equal(seq.X[i, :, 0], np.arange(p - L + 1, p + 1))   # no row after p
    assert seq.index[0] == sup.index[10]


def test_positions_without_a_full_history_are_dropped(ohlcv):
    sup = _sup(ohlcv).iloc[:30]
    scaled = np.zeros((30, 2))
    seq = build_sequences(scaled, sup, np.array([0, 1, 2, 5, 6]), seq_len=5)
    assert list(seq.index) == list(sup.index[[5, 6]])


def test_targets_come_from_the_prediction_day_row(ohlcv):
    sup = _sup(ohlcv).iloc[:40]
    scaled = np.zeros((40, 2))
    pos = np.arange(8, 20)
    seq = build_sequences(scaled, sup, pos, seq_len=4)
    assert np.allclose(seq.y_ret, sup["target_return"].to_numpy()[pos])
    assert np.array_equal(seq.y_dir, sup["target_direction"].to_numpy()[pos])


def test_split_sizes_and_look_back_into_earlier_data(ohlcv):
    sup = _sup(ohlcv)
    split = chronological_split(sup, 0.7, 0.15, gap=1)
    cols = feature_columns(sup, scale_free_only=True)
    L = 10
    data, _ = prepare_sequence_data(sup, split, cols, L)
    assert len(data["train"]) == len(split.train) - (L - 1)   # first rows lack history
    assert len(data["val"]) == len(split.val)                  # val reaches back into train features
    assert len(data["test"]) == len(split.test)
    assert data["val"].index.min() == split.val.index.min()    # labels/dates still belong to val


def test_scaler_uses_training_rows_only(ohlcv):
    sup = _sup(ohlcv)
    split = chronological_split(sup, 0.7, 0.15, gap=1)
    cols = feature_columns(sup, scale_free_only=True)
    _, scaler = prepare_sequence_data(sup, split, cols, 10, clip=0)
    assert np.allclose(scaler.scaler.mean_, split.train[cols].mean().to_numpy())


def test_changing_test_period_data_does_not_change_train_or_val_sequences(ohlcv):
    sup = _sup(ohlcv)
    split = chronological_split(sup, 0.7, 0.15, gap=1)
    cols = feature_columns(sup, scale_free_only=True)
    a, _ = prepare_sequence_data(sup, split, cols, 10)
    sup2 = sup.copy()
    sup2.loc[split.test.index, cols] = sup2.loc[split.test.index, cols] * 10.0
    b, _ = prepare_sequence_data(sup2, split, cols, 10)
    assert np.array_equal(a["train"].X, b["train"].X)
    assert np.array_equal(a["val"].X, b["val"].X)
    assert not np.array_equal(a["test"].X, b["test"].X)
