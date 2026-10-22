from pathlib import Path

import pytest

pytest.importorskip("plotly")
pytest.importorskip("xgboost")
pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _app(monkeypatch):
    monkeypatch.setenv("STOCKAI_SYNTHETIC", "1")          # no network: the app uses fake market data
    return AppTest.from_file(APP, default_timeout=120)


def test_dashboard_renders_without_exceptions(monkeypatch):
    at = _app(monkeypatch).run()
    assert not at.exception
    assert any("not financial advice" in str(w.value).lower() for w in at.warning)
    assert len(at.tabs) == 8


def test_invalid_ticker_is_rejected_in_the_sidebar(monkeypatch):
    at = _app(monkeypatch).run()
    at.sidebar.text_input[0].set_value("AA PL")
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("invalid ticker" in str(e.value).lower() for e in list(at.error) + list(at.sidebar.error))
