import json
import tempfile
from pathlib import Path

import numpy as np

from src.evaluation.compare import build_comparison, significant_models
from src.visualization.model_plots import plot_model_comparison


def _cls_row(model, acc, lo, hi, p, auc=None, alo=None, ahi=None):
    row = {"model": model, "accuracy": acc, "acc_CI95": f"[{lo:.3f}, {hi:.3f}]", "p_vs_majority": p,
           "roc_auc": auc, "brier": None}
    if alo is not None:
        row["auc_CI95"] = f"[{alo:.3f}, {ahi:.3f}]"
    return row


def _write(d: Path, tag: str, models):
    reg = [{"model": "previous_day", "RMSE": 4.5, "RMSE_vs_naive": 1.0, "oosR2_ret": 0.0, "dir_acc_%": float("nan")},
           {"model": f"{tag}_reg", "RMSE": 4.4, "RMSE_vs_naive": 0.99, "oosR2_ret": 0.003, "dir_acc_%": 52.0}]
    cls = [_cls_row("always_majority", 0.53, 0.48, 0.58, 0.5, 0.5, 0.5, 0.5)] + models
    (d / f"{tag}_results_AAPL.json").write_text(json.dumps(
        {"regression": {"test": reg}, "classification": {"test": cls}}))


def test_comparison_merges_files_and_dedupes_baselines():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        _write(d, "xgboost", [_cls_row("xgboost_raw", 0.54, 0.49, 0.59, 0.4, 0.526, 0.468, 0.583)])
        _write(d, "lstm", [_cls_row("lstm_raw", 0.50, 0.45, 0.55, 0.9, 0.48, 0.43, 0.53)])
        reg, cls, found = build_comparison(d, "AAPL")
    assert found == ["xgboost_results_AAPL.json", "lstm_results_AAPL.json"]
    assert list(cls.index).count("always_majority") == 1
    assert {"xgboost_raw", "lstm_raw"} <= set(cls.index)
    assert np.isclose(cls.loc["xgboost_raw", "auc_lo"], 0.468)
    assert "previous_day" in reg.index and "lstm_reg" in reg.index


def test_missing_files_are_skipped():
    with tempfile.TemporaryDirectory() as t:
        reg, cls, found = build_comparison(Path(t), "AAPL")
    assert found == [] and reg.empty and cls.empty


def test_significance_requires_both_p_value_and_auc_interval():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        _write(d, "xgboost", [
            _cls_row("good", 0.60, 0.55, 0.65, 0.001, 0.62, 0.56, 0.68),
            _cls_row("p_only", 0.60, 0.55, 0.65, 0.001, 0.52, 0.47, 0.57),
            _cls_row("auc_only", 0.53, 0.48, 0.58, 0.4, 0.60, 0.55, 0.65),
        ])
        _, cls, _ = build_comparison(d, "AAPL")
    assert significant_models(cls) == ["good"]


def test_comparison_plot_is_written():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        _write(d, "xgboost", [_cls_row("xgboost_raw", 0.54, 0.49, 0.59, 0.4, 0.526, 0.468, 0.583)])
        _, cls, _ = build_comparison(d, "AAPL")
        out = plot_model_comparison(cls, d / "cmp.png")
        assert Path(out).exists() and Path(out).stat().st_size > 1000
