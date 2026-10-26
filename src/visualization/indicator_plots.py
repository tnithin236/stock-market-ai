"""Plotly charts for the dashboard."""
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_BLUE, _DARK, _RED, _GREEN = "#3b82f6", "#0f172a", "#dc2626", "#16a34a"


def _layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(title=title, template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.12))
    return fig


def plot_price_bollinger(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], fill="tonexty", fillcolor="rgba(59,130,246,0.15)",
                             line=dict(width=0), name="Bollinger band (20, 2)"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color=_DARK, width=1.6)))
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20", line=dict(width=1.1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50", line=dict(width=1.1)))
    return _layout(fig, f"{ticker}: price, moving averages and Bollinger bands", 420)


def plot_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df.index, y=df["rsi_14"], name="RSI 14", line=dict(color=_BLUE, width=1.4)))
    fig.add_hline(y=70, line_dash="dot", line_color=_RED)
    fig.add_hline(y=30, line_dash="dot", line_color=_GREEN)
    fig.update_yaxes(range=[0, 100])
    return _layout(fig, "RSI (14): above 70 often called overbought, below 30 oversold", 300)


def plot_macd(df: pd.DataFrame) -> go.Figure:
    colors = [_GREEN if v >= 0 else _RED for v in df["macd_hist"].fillna(0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="Histogram", marker_color=colors, opacity=0.5))
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color=_BLUE, width=1.3)))
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="#f59e0b", width=1.3)))
    return _layout(fig, "MACD (12, 26, 9)", 300)


def plot_pred_vs_actual(frame: pd.DataFrame, label: str, n_last: int = 120) -> go.Figure:
    f = frame.tail(n_last)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f.index, y=f["actual_next_close"], name="Actual next-day close",
                             line=dict(color=_DARK, width=2)))
    fig.add_trace(go.Scatter(x=f.index, y=f["close"], name="Previous-day baseline",
                             line=dict(color="#94a3b8", width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(x=f.index, y=f["model_pred_close"], name=f"{label} prediction",
                             line=dict(color=_BLUE, width=1.4)))
    return _layout(fig, f"Actual vs predicted next-day close (last {len(f)} test days)", 400)


def plot_equity_curves(equity: pd.DataFrame, title: str = "Portfolio value") -> go.Figure:
    fig = go.Figure()
    for name in equity.columns:
        bh = name == "buy_and_hold"
        fig.add_trace(go.Scatter(x=equity.index, y=equity[name], name=name,
                                 line=dict(width=2.6 if bh else 1.3, color=_DARK if bh else None)))
    return _layout(fig, title, 420)


def plot_bar(series: pd.Series, title: str, top: int = 15, error: Optional[pd.Series] = None) -> go.Figure:
    s = series.head(top)[::-1]
    err = dict(type="data", array=error.reindex(s.index).to_numpy()) if error is not None else None
    fig = go.Figure(go.Bar(x=s.values, y=list(s.index), orientation="h", marker_color=_BLUE, error_x=err))
    return _layout(fig, title, max(260, 24 * len(s) + 110))


def plot_model_comparison_plotly(cls: pd.DataFrame) -> go.Figure:
    """Accuracy and ROC-AUC with 95% intervals for every model (dashed line = the bar to beat)."""
    d = cls[cls["roc_auc"].notna()]
    names = list(d.index)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Accuracy", "ROC-AUC"))
    for col, (val, lo, hi) in enumerate((("accuracy", "acc_lo", "acc_hi"), ("roc_auc", "auc_lo", "auc_hi")), start=1):
        v = d[val].to_numpy(dtype=float)
        err_hi = (d[hi].to_numpy(dtype=float) - v).clip(min=0)
        err_lo = (v - d[lo].to_numpy(dtype=float)).clip(min=0)
        fig.add_trace(go.Scatter(x=v, y=names, mode="markers", marker=dict(color=_BLUE, size=9), showlegend=False,
                                 error_x=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo,
                                              color="#94a3b8")), row=1, col=col)
    ref = float(d.loc["always_majority", "accuracy"]) if "always_majority" in d.index else 0.5
    fig.add_vline(x=ref, line_dash="dot", line_color=_RED, row=1, col=1)
    fig.add_vline(x=0.5, line_dash="dot", line_color=_RED, row=1, col=2)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Test-set model comparison (95% intervals)", max(300, 34 * len(names) + 140))
