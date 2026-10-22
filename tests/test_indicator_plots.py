import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from src.dashboard.data import add_indicators  # noqa: E402
from src.data.synthetic import make_ohlcv  # noqa: E402
from src.visualization import indicator_plots as ip  # noqa: E402


def test_indicator_figures_build():
    d = add_indicators(make_ohlcv(300))
    assert len(ip.plot_price_bollinger(d, "TST").data) == 5
    assert len(ip.plot_rsi(d).data) == 1
    assert len(ip.plot_macd(d).data) == 3


def test_prediction_equity_and_bar_figures():
    idx = pd.bdate_range("2025-01-01", periods=50)
    frame = pd.DataFrame({"close": 100.0, "actual_next_close": 101.0, "model_pred_close": 100.5}, index=idx)
    assert len(ip.plot_pred_vs_actual(frame, "XGBoost", n_last=20).data) == 3
    eq = pd.DataFrame({"buy_and_hold": np.linspace(1, 1.2, 50), "x": np.linspace(1, 1.1, 50)}, index=idx)
    assert len(ip.plot_equity_curves(eq).data) == 2
    s = pd.Series([0.3, 0.2, 0.1], index=["a", "b", "c"])
    assert len(ip.plot_bar(s, "t", error=pd.Series([0.01, 0.02, 0.03], index=s.index)).data) == 1
