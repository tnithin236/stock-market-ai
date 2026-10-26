"""Streamlit dashboard.   Run from the project root:   streamlit run app/streamlit_app.py

EXPERIMENTAL RESEARCH / EDUCATIONAL USE ONLY - NOT FINANCIAL ADVICE.
"""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.backtesting.runner import run_backtest_study  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402
from src.dashboard import ui  # noqa: E402
from src.dashboard.data import add_indicators, get_market_data, kpi_series, summary_stats  # noqa: E402
from src.dashboard.explain_view import xgb_importance  # noqa: E402
from src.dashboard.results import (MODEL_LABELS, direction_summary, load_results, permutation_table,  # noqa: E402
                                   prediction_frame, results_tables)
from src.data.loader import normalize_ticker  # noqa: E402
from src.evaluation.compare import build_comparison, significant_models  # noqa: E402
from src.features.engineering import build_features  # noqa: E402
from src.serving.errors import (DataUnavailableError, InsufficientHistoryError, ModelNotTrainedError)  # noqa: E402
from src.serving.service import DISCLAIMER, PredictionService  # noqa: E402
from src.visualization import indicator_plots as ip  # noqa: E402
from src.visualization.eda_plots import plot_candlestick  # noqa: E402

st.set_page_config(page_title="Stock Market Prediction & Analysis", page_icon="📈", layout="wide")
st.markdown(f"<style>{ui.APP_CSS}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def html(block: str) -> None:
    st.markdown(block, unsafe_allow_html=True)


def card():
    """A bordered container (falls back to a plain one on older Streamlit versions)."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def show_plot(fig) -> None:
    fig = ui.style_dark(fig)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                                   # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True)


def show_table(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


def wide_submit(text: str) -> bool:
    """Full-width form button across Streamlit versions (newest API first)."""
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            return st.form_submit_button(text, **kwargs)
        except TypeError:
            continue
    return False


@st.cache_resource
def get_cfg():
    return load_config()


@st.cache_resource
def get_service() -> PredictionService:
    return PredictionService.from_config()


@st.cache_data(show_spinner="Loading market data...", ttl=3600)
def load_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    return get_market_data(ticker, start, end, get_cfg())


cfg = get_cfg()
svc = get_service()
reports_dir = resolve_path(cfg["paths"].get("reports_dir", "reports"))
models_dir = resolve_path(cfg["paths"]["models_dir"])

# ----------------------------------------------------------------------------- top bar
bar_left, bar_right = st.columns([5, 3])
with bar_left:
    html(ui.brand_html())
with bar_right:
    page = st.radio("Navigate", ["Dashboard", "About"], horizontal=True, label_visibility="collapsed", key="page")

# ----------------------------------------------------------------------------- sidebar
DEFAULTS = {"ticker": "AAPL", "start": date(2015, 1, 1), "end": date.today(), "model": "xgboost"}
if "params" not in st.session_state:
    st.session_state["params"] = dict(DEFAULTS)
P = st.session_state["params"]

with st.sidebar:
    html(ui.sidebar_header_html())
    with st.form("controls"):
        ticker_in = st.text_input("Ticker symbol", value=P["ticker"], help="e.g. AAPL, MSFT, ^GSPC")
        dates_in = st.date_input("Date range", value=(P["start"], P["end"]),
                                 min_value=date(2000, 1, 1), max_value=date.today())
        model_in = st.selectbox("Model", list(MODEL_LABELS), index=list(MODEL_LABELS).index(P["model"]),
                                format_func=lambda k: MODEL_LABELS[k])
        submitted = wide_submit("🔄  Load / refresh")
    if submitted:
        try:
            t_ok = normalize_ticker(ticker_in)
            if not isinstance(dates_in, (tuple, list)) or len(dates_in) != 2:
                raise ValueError("Pick both a start and an end date.")
            if dates_in[0] >= dates_in[1]:
                raise ValueError("The start date must be before the end date.")
            st.session_state["params"] = {"ticker": t_ok, "start": dates_in[0], "end": dates_in[1], "model": model_in}
            P = st.session_state["params"]
        except ValueError as exc:
            st.error(str(exc))
    st.caption(DISCLAIMER)

ticker, model = P["ticker"], P["model"]
label = MODEL_LABELS[model]

# ----------------------------------------------------------------------------- hero + disclaimer
html(ui.hero_html("Leakage-safe machine-learning research on next-day stock forecasting"))
html(ui.banner_html("Experimental research / educational system - not financial advice.",
                    "In this project's held-out tests, no model showed a statistically significant edge over simple baselines."))

if page == "About":
    st.markdown(ui.about_markdown())
    st.stop()

# ----------------------------------------------------------------------------- data + KPI cards
try:
    df = load_data(ticker, P["start"], P["end"])
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load data for {ticker}: {exc}")
    st.stop()
if len(df) < 60:
    st.error("Not enough data in this range (need at least ~60 trading days). Widen the date range.")
    st.stop()

ind_df = add_indicators(df)
stats = summary_stats(df)
spark = kpi_series(df)
up_day = stats["last_day_change_pct"] >= 0
period_up = stats["period_return_pct"] >= 0
k1, k2, k3, k4 = st.columns(4)
with k1:
    html(ui.kpi_card_html(f"{ticker} latest close", f"${stats['latest_close']:,.2f}", "🪙", ui.PURPLE,
                          f"{stats['last_day_change_pct']:+.2f}% last day", up_day, spark["close"]))
with k2:
    html(ui.kpi_card_html("Period return", f"{stats['period_return_pct']:+,.1f}%", "📅", ui.GREEN if period_up else ui.RED,
                          None, None, spark["cum_return_pct"], ui.GREEN if period_up else ui.RED,
                          "green" if period_up else "red"))
with k3:
    html(ui.kpi_card_html("Annualized volatility", f"{stats['annualized_volatility_pct']:.1f}%", "〰️", ui.CYAN,
                          None, None, spark["volatility_pct"]))
with k4:
    html(ui.kpi_card_html("Max drawdown (period)", f"{stats['max_drawdown_pct']:.1f}%", "📉", ui.PURPLE,
                          None, None, spark["drawdown_pct"], ui.PURPLE))
st.write("")

tabs = st.tabs(["📊 Market data", "🧭 Indicators", "🔮 Prediction", "📈 Actual vs predicted", "🧮 Metrics",
                "🧠 Feature importance", "🧪 Backtest", "⚖️ Compare models"])

# ----------------------------------------------------------------------------- 1. market data
with tabs[0]:
    with card():
        show_plot(plot_candlestick(df, ticker, (20, 50, 200)))
    st.download_button("⬇️  Download this data as CSV", data=df.to_csv().encode("utf-8"),
                       file_name=f"{ticker}_{P['start']}_{P['end']}.csv", mime="text/csv")
    st.caption("Prices are split/dividend adjusted (Yahoo Finance via yfinance). Latest rows:")
    show_table(df.tail(15).round(3))

# ----------------------------------------------------------------------------- 2. indicators
with tabs[1]:
    with card():
        show_plot(ip.plot_price_bollinger(ind_df, ticker))
    left, right = st.columns(2)
    with left:
        with card():
            show_plot(ip.plot_rsi(ind_df))
    with right:
        with card():
            show_plot(ip.plot_macd(ind_df))
    with st.expander("What do these indicators mean?"):
        st.markdown(
            "- **SMA 20/50**: average price over 20/50 days; smooths noise and shows trend.\n"
            "- **Bollinger bands**: 20-day average +/- 2 standard deviations; wide bands mean high volatility.\n"
            "- **RSI (14)**: 0-100 momentum gauge; >70 is called overbought, <30 oversold.\n"
            "- **MACD**: gap between 12- and 26-day exponential averages, with a 9-day signal line.\n\n"
            "All indicators use only past data (no look-ahead).")

# ----------------------------------------------------------------------------- 3. prediction
with tabs[2]:
    st.subheader(f"{label} prediction for the next trading session")
    avail = svc.available_models(ticker)
    if model in avail:
        if st.button("Generate prediction", type="primary"):
            try:
                with st.spinner("Fetching the latest data and running the model..."):
                    st.session_state["last_pred"] = (ticker, model, svc.predict(ticker, model))
            except (ModelNotTrainedError, DataUnavailableError, InsufficientHistoryError, ValueError) as exc:
                st.session_state.pop("last_pred", None)
                st.error(str(exc))
        stored = st.session_state.get("last_pred")
        if stored and stored[0] == ticker and stored[1] == model:
            res = stored[2]
            a, b, c, d = st.columns(4)
            a.metric("Direction", res["prediction"])
            b.metric("Predicted close", f"${res['predicted_close']:,.2f}", f"{res['predicted_return_pct']:+.2f}%")
            c.metric("Raw score (NOT a probability)", f"{res['score']:.3f}")
            if res.get("calibrated_probability_up") is not None:
                d.metric("Calibrated P(up)", f"{res['calibrated_probability_up']:.3f}")
            st.caption(f"As of {res['as_of_date']} close. " + res["calibration_note"])
            ev = res.get("evidence")
            if ev:
                st.markdown("**Held-out evidence for this model**")
                show_table(pd.DataFrame({
                    "Test accuracy": [ev.get("test_accuracy")], "Always-UP accuracy": [ev.get("always_up_accuracy")],
                    "ROC-AUC": [ev.get("roc_auc")], "AUC 95% interval": [ev.get("roc_auc_95ci")],
                    "p-value vs always-UP": [ev.get("p_vs_majority")],
                    "Significant edge?": ["yes" if ev.get("significant_edge") else "no"]}))
            for w in res["warnings"]:
                st.warning(w)
    else:
        st.info(f"No trained {label} model found for {ticker}.")
        if model == "xgboost":
            if st.button("Train XGBoost for this ticker (about 30-60 seconds)"):
                try:
                    with st.spinner("Training... (downloads data, builds features, trains and evaluates)"):
                        svc.trainer(ticker, cfg["data"]["start_date"], None)
                    st.success("Training finished.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Training failed: {exc}")
        else:
            st.code(f"python main.py {model} --ticker {ticker}", language="powershell")
            st.caption("Run this in a terminal, then refresh this page.")

# ----------------------------------------------------------------------------- 4. actual vs predicted
with tabs[3]:
    frame = prediction_frame(reports_dir, ticker, model, df)
    if frame is None:
        st.info(f"No saved test-period predictions for {ticker} / {label} in this date range. "
                f"Run `python main.py {model} --ticker {ticker}` (and keep the test period inside the date range).")
    else:
        with card():
            show_plot(ip.plot_pred_vs_actual(frame, label))
        s = direction_summary(frame)
        a, b, c = st.columns(3)
        a.metric("Direction accuracy (score > 0.5)", f"{s['accuracy'] * 100:.1f}%")
        b.metric("Always-UP accuracy", f"{s['always_up_accuracy'] * 100:.1f}%")
        c.metric("Test days shown", s["n_days"])
        st.caption("Price forecasts are Close x (1 + predicted return). The prediction hugs the previous-day line because "
                   "daily returns are mostly unpredictable; compare errors, not appearance.")

# ----------------------------------------------------------------------------- 5. metrics
with tabs[4]:
    results = load_results(reports_dir, ticker, model)
    if results is None:
        st.info(f"No saved evaluation for {ticker} / {label}. Run `python main.py {model} --ticker {ticker}`.")
    else:
        reg_df, cls_df = results_tables(results)
        st.caption(f"Held-out TEST period: {' to '.join(results.get('test_period') or ['unknown'])}")
        ev = svc.evidence(ticker, model)
        if ev:
            a, b, c, d = st.columns(4)
            a.metric("Test accuracy", f"{ev['test_accuracy'] * 100:.1f}%")
            b.metric("Always-UP accuracy", f"{ev['always_up_accuracy'] * 100:.1f}%")
            c.metric("ROC-AUC", f"{ev['roc_auc']:.3f}", ev.get("roc_auc_95ci") or None, delta_color="off")
            d.metric("Significant edge?", "yes" if ev["significant_edge"] else "no")
        st.markdown("**Regression: next-day close** (RMSE_vs_naive < 1 and oosR2_ret > 0 would mean an edge)")
        show_table(reg_df.round(4))
        st.markdown("**Classification: next-day direction**")
        show_table(cls_df.round(4))

# ----------------------------------------------------------------------------- 6. feature importance
with tabs[5]:
    if model == "xgboost":
        imp = xgb_importance(models_dir, ticker, build_features(df, cfg.get("features")))
        if imp is None:
            st.info(f"No trained XGBoost model for {ticker}.")
        else:
            left, right = st.columns(2)
            with left:
                with card():
                    show_plot(ip.plot_bar(imp["gain"], "Classifier: gain importance"))
            with right:
                if imp["shap"] is not None:
                    with card():
                        show_plot(ip.plot_bar(imp["shap"], "Classifier: mean |SHAP| (recent rows)"))
                else:
                    st.info("SHAP values unavailable for this model.")
    else:
        perm = permutation_table(load_results(reports_dir, ticker, model) or {}, "classifier")
        if perm is None:
            st.info(f"No saved permutation importance for {ticker} / {label}.")
        else:
            with card():
                show_plot(ip.plot_bar(perm["importance_mean"], f"{label} classifier: permutation importance (AUC drop)",
                                      error=perm["importance_std"]))
    st.caption("Importance shows what a model USED, not what is causally true. Near-random targets give unstable "
               "rankings; correlated features share credit; permutation creates unrealistic inputs.")

# ----------------------------------------------------------------------------- 7. backtest
with tabs[6]:
    st.markdown("Long/flat strategy on the held-out TEST period. Signal at the close, trade at the next open. "
                "Costs are charged on turnover. Simulated results are not predictive.")
    col_a, col_b = st.columns(2)
    cost = col_a.slider("One-way trading cost (basis points)", 0, 50, 5)
    execution = col_b.selectbox("Execution", ["next_open", "same_close"])
    if st.button("Run backtest"):
        try:
            with st.spinner("Running backtest..."):
                st.session_state["bt"] = (ticker, run_backtest_study(
                    cfg, ticker, cost_bps=float(cost), execution=execution, make_plots=False, save=False))
        except (ValueError, FileNotFoundError, KeyError) as exc:
            st.session_state.pop("bt", None)
            st.error(f"{exc}")
    bt = st.session_state.get("bt")
    if bt and bt[0] == ticker:
        out = bt[1]
        equity = pd.DataFrame({n: c["equity"] for n, c in out["curves"].items()})
        with card():
            show_plot(ip.plot_equity_curves(equity, f"{ticker}: portfolio value after {cost} bps costs"))
        cols = ["total_return_%", "CAGR_%", "sharpe", "max_drawdown_%", "win_rate_days_%", "n_trades", "exposure_%",
                "cost_drag_%"]
        show_table(out["metrics"][cols].round(2))
        st.markdown("**Total return % by one-way cost (bps)**")
        show_table(out["cost_returns"].round(2))
        st.markdown("**Better than luck?** (random strategies with the same average exposure)")
        show_table(out["null"].round(3))
    else:
        st.info("Click **Run backtest**. It needs saved predictions: run `python main.py xgboost`, `lstm` and "
                "`transformer` first.")

# ----------------------------------------------------------------------------- 8. compare models
with tabs[7]:
    reg_c, cls_c, found = build_comparison(reports_dir, ticker)
    if not found:
        st.info("No result files yet. Run the model commands first.")
    else:
        st.caption("Test-period results from: " + ", ".join(found))
        st.markdown("**Regression**")
        show_table(reg_c.round(4))
        st.markdown("**Classification**")
        show_table(cls_c[["accuracy", "p_vs_majority", "roc_auc", "auc_lo", "auc_hi", "brier"]].round(4))
        winners = significant_models(cls_c)
        if winners:
            st.success(f"Statistically significant (p<0.05 and AUC interval above 0.5): {winners}")
        else:
            st.info("No model shows a statistically significant edge over the always-UP baseline.")
        with card():
            show_plot(ip.plot_model_comparison_plotly(cls_c))

st.divider()
st.caption(DISCLAIMER)
