"""HTML/CSS building blocks for the dashboard (no Streamlit import, so they are unit-testable).

Selectors in APP_CSS target Streamlit's own markup; if a future version renames them the app still works, it just
falls back to the plain dark theme from .streamlit/config.toml.
"""
import html
from typing import Any, Iterable, Optional

import numpy as np

CYAN, PURPLE, GREEN, RED, MUTED = "#22d3ee", "#8b5cf6", "#22c55e", "#ef4444", "#94a3b8"
_DARK_LINES = {"#0f172a", "#0f1729"}


def sparkline_svg(values: Iterable[float], color: str = CYAN, width: int = 150, height: int = 46) -> str:
    """Tiny inline line chart. Returns '' when there is nothing sensible to draw."""
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return ""
    lo, hi = float(v.min()), float(v.max())
    span = (hi - lo) or 1.0
    xs = np.linspace(2, width - 2, len(v))
    ys = (height - 4) - (v - lo) / span * (height - 8)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"2,{height - 2} {pts} {width - 2},{height - 2}"
    gid = f"sg{abs(sum(map(ord, color))) + len(v)}"
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity="0.35"/><stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
            f'</linearGradient></defs><polygon points="{area}" fill="url(#{gid})"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/></svg>')


def kpi_card_html(title: str, value: str, icon: str = "●", accent: str = CYAN, delta: Optional[str] = None,
                  delta_positive: Optional[bool] = None, spark: Optional[Iterable[float]] = None,
                  value_color: Optional[str] = None, tint: Optional[str] = None) -> str:
    """One KPI card. All text is HTML-escaped."""
    delta_html = ""
    if delta:
        cls = "pos" if delta_positive else ("neg" if delta_positive is False else "neu")
        arrow = "▲" if delta_positive else ("▼" if delta_positive is False else "")
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {html.escape(delta)}</div>'
    spark_html = f'<div class="kpi-spark">{sparkline_svg(spark, accent)}</div>' if spark is not None else ""
    style = f' style="color:{value_color}"' if value_color else ""
    tint_cls = f" kpi-tint-{tint}" if tint else ""
    return (f'<div class="kpi-card{tint_cls}"><div class="kpi-top"><span class="kpi-icon" style="background:{accent}22;color:{accent}">'
            f'{html.escape(icon)}</span><span class="kpi-title">{html.escape(title)}</span></div>'
            f'<div class="kpi-value"{style}>{html.escape(value)}</div>{delta_html}{spark_html}</div>')


def brand_html() -> str:
    return ('<div class="brand"><span class="brand-logo">📈</span><span class="brand-text">Stock Market '
            '<span class="grad">Prediction &amp; Analysis</span></span>'
            '<span class="brand-pill">Research • Educational</span></div>')


def hero_html(subtitle: str) -> str:
    bars = "".join(
        f'<rect x="{10 + i * 18}" y="{60 - h}" width="9" height="{h}" rx="2" fill="{c}" opacity="0.85"/>'
        for i, (h, c) in enumerate([(14, RED), (22, GREEN), (30, GREEN), (24, RED), (36, GREEN), (44, GREEN), (38, RED), (52, GREEN)]))
    deco = (f'<svg class="hero-deco" viewBox="0 0 170 70" xmlns="http://www.w3.org/2000/svg">{bars}'
            f'<polyline points="8,60 40,50 70,44 100,30 130,24 160,6" fill="none" stroke="{PURPLE}" stroke-width="2.5" stroke-linecap="round"/></svg>')
    return ('<div class="hero"><div class="hero-icon">📊</div><div class="hero-body">'
            '<div class="hero-title">Stock Market <span class="grad">Prediction &amp; Analysis</span></div>'
            f'<div class="hero-sub">{html.escape(subtitle)}</div></div>{deco}</div>')


def banner_html(headline: str, body: str, icon: str = "💡") -> str:
    return (f'<div class="banner"><div class="banner-icon">{html.escape(icon)}</div><div>'
            f'<div class="banner-head">{html.escape(headline)}</div><div class="banner-body">{html.escape(body)}</div></div></div>')


def sidebar_header_html() -> str:
    return ('<div class="side-head"><span class="side-gear">⚙️</span><div><div class="side-title">Controls</div>'
            '<div class="side-sub">Adjust parameters and select your settings</div></div></div>')


def style_dark(fig: Any) -> Any:
    """Make a Plotly figure match the dark UI: transparent background, soft grid, light text, light price lines."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e5e7eb"))
    try:
        fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
        fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    except Exception:  # noqa: BLE001 - styling must never break a chart
        pass
    for trace in getattr(fig, "data", []):
        line = getattr(trace, "line", None)
        if line is not None and getattr(line, "color", None) in _DARK_LINES:
            line.color = "#e2e8f0"
    return fig


def about_markdown() -> str:
    return """
### About this project
An end-to-end research system that tries to forecast next-day stock price and direction with **baselines, XGBoost, an LSTM and a
Transformer**, evaluates them with leakage-safe time-series methodology, and backtests the resulting signals with trading costs.

**What the evidence says.** Across a single held-out test, 15 walk-forward blocks and 10 US large-cap stocks, **no model showed a
statistically significant edge** over simple baselines, and no strategy beat buy-and-hold after costs once multiple comparisons
were accounted for. The project's value is the methodology: leakage prevention, honest statistics and a tested, deployable system.

**How it is built**
- Causal technical features, chronological splits with a gap, train-only scaling
- Early stopping, calibration, SHAP and permutation importance
- Cost-aware backtesting with a random-strategy luck test
- FastAPI backend, this Streamlit dashboard, Docker and CI

**Use the tabs on the Dashboard page** to explore data, indicators, predictions (with their held-out evidence), metrics,
feature importance, backtests and the model comparison. See the repository README for full results and limitations.

> **Experimental research / educational system. Not financial advice.** Do not trade on its output.
"""


APP_CSS = """
.stApp { background: radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.18), transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,0.10), transparent 55%), #070b1f; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1500px; }
.grad { background: linear-gradient(90deg, #22d3ee, #8b5cf6); -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent; color: transparent; }

/* top bar */
.brand { display:flex; align-items:center; gap:12px; padding:6px 2px; }
.brand-logo { font-size:1.6rem; }
.brand-text { font-weight:700; font-size:1.15rem; color:#f1f5f9; }
.brand-pill { margin-left:auto; padding:6px 14px; border-radius:999px; font-size:0.78rem; color:#cbd5e1;
              background:rgba(30,41,90,0.7); border:1px solid rgba(99,102,241,0.35); }
div[data-testid="stRadio"] > div { gap: 6px; justify-content: flex-end; }
div[data-testid="stRadio"] label { background: rgba(15,24,56,0.7); border:1px solid rgba(99,102,241,0.25);
                                    padding: 6px 16px; border-radius: 10px; }

/* hero + banner */
.hero { display:flex; align-items:center; gap:22px; padding:22px 28px; border-radius:18px; margin:6px 0 14px 0;
        background: linear-gradient(135deg, rgba(20,30,80,0.85), rgba(10,16,48,0.85)); border:1px solid rgba(99,102,241,0.35);
        box-shadow: 0 0 40px rgba(99,102,241,0.12); }
.hero-icon { width:68px; height:68px; border-radius:18px; display:flex; align-items:center; justify-content:center; font-size:2rem;
             background: linear-gradient(135deg, #6366f1, #8b5cf6); box-shadow: 0 8px 24px rgba(99,102,241,0.45); }
.hero-title { font-size:2.1rem; font-weight:800; color:#f8fafc; line-height:1.15; }
.hero-sub { color:#94a3b8; margin-top:6px; }
.hero-deco { margin-left:auto; width:170px; height:70px; opacity:0.9; }
.banner { display:flex; gap:16px; align-items:center; padding:14px 20px; border-radius:14px; margin-bottom:16px;
          background: linear-gradient(90deg, rgba(139,92,246,0.18), rgba(34,211,238,0.06)); border:1px solid rgba(34,211,238,0.25);
          border-left:4px solid #8b5cf6; }
.banner-icon { font-size:1.4rem; width:44px; height:44px; border-radius:50%; display:flex; align-items:center; justify-content:center;
               background: rgba(15,24,56,0.9); border:1px solid rgba(99,102,241,0.4); }
.banner-head { color:#22d3ee; font-weight:700; }
.banner-body { color:#cbd5e1; font-size:0.92rem; }

/* sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(14,22,60,0.95), rgba(8,13,36,0.95));
                                     border-right:1px solid rgba(99,102,241,0.25); }
.side-head { display:flex; gap:12px; align-items:center; margin-bottom:10px; }
.side-gear { font-size:1.5rem; }
.side-title { font-size:1.35rem; font-weight:800; color:#f1f5f9; }
.side-sub { color:#94a3b8; font-size:0.85rem; }
div[data-testid="stForm"] { background: rgba(15,24,56,0.55); border:1px solid rgba(99,102,241,0.28); border-radius:14px; padding:14px; }
div[data-testid="stFormSubmitButton"] button, .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #06b6d4, #8b5cf6); color:white; border:0; border-radius:12px; font-weight:700; }

/* KPI cards */
.kpi-card { position:relative; padding:16px 18px 10px 18px; border-radius:16px; min-height:150px; overflow:hidden;
            background: linear-gradient(145deg, rgba(20,30,80,0.8), rgba(10,16,48,0.85)); border:1px solid rgba(99,102,241,0.3); }
.kpi-tint-green { background: linear-gradient(145deg, rgba(16,70,70,0.75), rgba(8,30,40,0.85)); border-color: rgba(34,197,94,0.4); }
.kpi-tint-red { background: linear-gradient(145deg, rgba(80,24,50,0.75), rgba(30,10,30,0.85)); border-color: rgba(239,68,68,0.4); }
.kpi-top { display:flex; align-items:center; gap:10px; }
.kpi-icon { width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:0.95rem; }
.kpi-title { color:#cbd5e1; font-size:0.9rem; }
.kpi-value { font-size:2rem; font-weight:800; color:#f8fafc; margin-top:8px; }
.kpi-delta { display:inline-block; margin-top:2px; padding:2px 10px; border-radius:999px; font-size:0.8rem; }
.kpi-delta.pos { color:#22c55e; background:rgba(34,197,94,0.15); }
.kpi-delta.neg { color:#ef4444; background:rgba(239,68,68,0.15); }
.kpi-delta.neu { color:#94a3b8; }
.kpi-spark { position:absolute; right:12px; bottom:8px; opacity:0.95; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:4px; background: rgba(15,24,56,0.6); border:1px solid rgba(99,102,241,0.25);
                                     border-radius:14px; padding:6px; }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:8px 14px; color:#94a3b8; }
.stTabs [aria-selected="true"] { background: rgba(34,211,238,0.12); color:#22d3ee !important; }
.stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #22d3ee, #8b5cf6); }

/* chart cards and metrics */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; border:1px solid rgba(99,102,241,0.28) !important;
                                                  background: rgba(10,16,48,0.55); }
div[data-testid="stMetric"] { background: rgba(15,24,56,0.6); border:1px solid rgba(99,102,241,0.25); border-radius:14px; padding:12px 16px; }
div[data-testid="stDataFrame"] { border-radius:12px; }
"""
