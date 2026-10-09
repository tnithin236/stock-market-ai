"""Interactive Plotly figures for EDA."""
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import acf

_WEEKENDS = dict(bounds=["sat", "mon"])


def plot_candlestick(df: pd.DataFrame, ticker: str, ma_windows: Iterable[int] = (20, 50, 200)) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC"), row=1, col=1)
    for w in ma_windows:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(w).mean(),
                                 mode="lines", name=f"SMA {w}", line=dict(width=1.3)),
                      row=1, col=1)
    colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume"),
                  row=2, col=1)
    fig.update_xaxes(rangebreaks=[_WEEKENDS])
    fig.update_layout(title=f"{ticker} price, moving averages and volume",
                      xaxis_rangeslider_visible=False, template="plotly_white",
                      height=750, legend=dict(orientation="h", y=1.02))
    return fig


def plot_returns_distribution(returns: pd.Series, ticker: str) -> go.Figure:
    r = returns.dropna() * 100
    grid = np.linspace(r.min(), r.max(), 400)
    normal = stats.norm.pdf(grid, r.mean(), r.std())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=120, histnorm="probability density",
                               name="Daily returns", opacity=0.7))
    fig.add_trace(go.Scatter(x=grid, y=normal, mode="lines", name="Normal fit",
                             line=dict(color="crimson", width=2)))
    fig.update_layout(title=f"{ticker} daily return distribution (fat tails vs. normal)",
                      xaxis_title="Daily return (%)", yaxis_title="Density",
                      template="plotly_white", height=450)
    return fig


def plot_rolling_volatility(returns: pd.Series, ticker: str, window: int = 21) -> go.Figure:
    vol = returns.rolling(window).std() * np.sqrt(252) * 100
    fig = go.Figure(go.Scatter(x=vol.index, y=vol, mode="lines", name=f"{window}d vol"))
    fig.update_layout(title=f"{ticker} rolling {window}-day annualized volatility (regimes)",
                      yaxis_title="Volatility (%)", template="plotly_white", height=400)
    return fig


def plot_acf_comparison(returns: pd.Series, ticker: str, nlags: int = 30) -> go.Figure:
    """ACF of returns vs squared returns: returns ~uncorrelated, volatility is not."""
    r = returns.dropna()
    a_ret = acf(r, nlags=nlags)[1:]
    a_sq = acf(r ** 2, nlags=nlags)[1:]
    lags = list(range(1, nlags + 1))
    band = 1.96 / np.sqrt(len(r))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF of returns", "ACF of squared returns"))
    fig.add_trace(go.Bar(x=lags, y=a_ret, name="returns"), row=1, col=1)
    fig.add_trace(go.Bar(x=lags, y=a_sq, name="squared returns"), row=1, col=2)
    for col in (1, 2):
        fig.add_hrect(y0=-band, y1=band, fillcolor="grey", opacity=0.2, line_width=0, row=1, col=col)
    fig.update_layout(title=f"{ticker} autocorrelation (shaded = 95% noise band)",
                      template="plotly_white", height=400, showlegend=False)
    return fig


def plot_price_vs_returns(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Price (non-stationary) next to log returns (roughly stationary)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Close price", "Log returns"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["log_return"], name="Log return",
                             line=dict(width=0.8)), row=2, col=1)
    fig.update_layout(title=f"{ticker}: why we model returns, not raw prices",
                      template="plotly_white", height=600, showlegend=False)
    return fig
