import re
from types import SimpleNamespace

import numpy as np

from src.dashboard import ui
from src.dashboard.data import kpi_series
from src.data.synthetic import make_ohlcv


def test_sparkline_handles_normal_constant_nan_and_empty_input():
    svg = ui.sparkline_svg([1, 2, 3, 2, 4], "#22d3ee")
    assert svg.startswith("<svg") and "polyline" in svg
    assert "polyline" in ui.sparkline_svg([5, 5, 5, 5])                        # flat line must not divide by zero
    assert ui.sparkline_svg([]) == "" and ui.sparkline_svg([1.0]) == ""
    clean = ui.sparkline_svg([1, np.nan, 3, np.inf, 2])
    assert "nan" not in clean.lower() and "inf" not in clean.lower()


def test_sparkline_coordinates_stay_inside_the_box():
    svg = ui.sparkline_svg(np.random.default_rng(0).normal(size=80), width=150, height=46)
    nums = [float(v) for v in re.findall(r"polyline points=\"([^\"]+)\"", svg)[0].replace(",", " ").split()]
    xs, ys = nums[0::2], nums[1::2]
    assert min(xs) >= 0 and max(xs) <= 150 and min(ys) >= 0 and max(ys) <= 46


def test_kpi_card_escapes_html_and_marks_direction():
    card = ui.kpi_card_html("<b>x</b>", "$1<script>", "i", delta="+1%", delta_positive=True, spark=[1, 2, 3])
    assert "<script>" not in card and "&lt;script&gt;" in card and "&lt;b&gt;" in card
    assert 'kpi-delta pos' in card and "▲" in card and "<svg" in card
    neg = ui.kpi_card_html("t", "v", delta="-1%", delta_positive=False)
    assert 'kpi-delta neg' in neg and "▼" in neg and "<svg" not in neg
    assert "kpi-tint-green" in ui.kpi_card_html("t", "v", tint="green")


def test_banners_and_headers_contain_the_honest_wording():
    assert "not financial advice" in ui.banner_html("Experimental research / educational system - not financial advice.", "x")
    page = ui.hero_html("Leakage-safe research") + ui.brand_html() + ui.sidebar_header_html()
    assert "Leakage-safe research" in page and "Controls" in page
    assert "investment decisions" not in page.lower() and "smarter predictions" not in page.lower()
    assert "not financial advice" in ui.about_markdown().lower()


def test_style_dark_recolors_dark_lines_and_sets_transparent_background():
    calls = {}
    dark = SimpleNamespace(line=SimpleNamespace(color="#0f172a"))
    blue = SimpleNamespace(line=SimpleNamespace(color="#3b82f6"))

    class Fig:
        data = [dark, blue]
        def update_layout(self, **kw): calls.update(kw)
        def update_xaxes(self, **kw): pass
        def update_yaxes(self, **kw): pass

    ui.style_dark(Fig())
    assert dark.line.color == "#e2e8f0" and blue.line.color == "#3b82f6"
    assert calls["template"] == "plotly_dark" and calls["paper_bgcolor"] == "rgba(0,0,0,0)"


def test_css_has_the_main_component_rules():
    for needle in (".kpi-card", ".hero", ".banner", 'data-baseweb="tab"', "stSidebar", "stMetric"):
        assert needle in ui.APP_CSS


def test_kpi_series_are_short_finite_and_aligned_with_the_data():
    df = make_ohlcv(500)
    s = kpi_series(df, n_points=60)
    assert set(s) == {"close", "cum_return_pct", "volatility_pct", "drawdown_pct"}
    for v in s.values():
        assert 2 <= len(v) <= 60 and np.isfinite(v).all()
    assert s["drawdown_pct"].max() <= 0 and abs(s["cum_return_pct"][0]) < 1e-9
