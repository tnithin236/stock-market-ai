import numpy as np

from src.evaluation.permutation import permutation_importance


def _neg_rmse(y, s):
    return -float(np.sqrt(np.mean((s - y) ** 2)))


def test_the_feature_the_model_uses_ranks_first_and_others_are_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5, 4))
    y = 2 * X[:, -1, 0] + rng.normal(0, 0.1, 300)
    imp = permutation_importance(lambda A: 2 * A[:, -1, 0], X, y, _neg_rmse,
                                 n_repeats=3, seed=1, feature_names=list("abcd"))
    assert imp.index[0] == "a" and imp.loc["a", "importance_mean"] > 0.5
    assert np.abs(imp.loc[["b", "c", "d"], "importance_mean"]).max() < 1e-12


def test_input_array_is_not_modified():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3, 2))
    before = X.copy()
    permutation_importance(lambda A: A[:, -1, 0], X, X[:, -1, 0], _neg_rmse, n_repeats=2)
    assert np.array_equal(X, before)
