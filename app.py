# ── AI Stock Screener ─────────────────────────────────────────
# Main Streamlit Application
# Built by Khan Atif — github.com/Khan7Atif9

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.insert(0, ".")

from config import TICKERS, APP_TITLE, APP_ICON
from data.fetcher import get_ohlcv, get_fundamentals
from utils.indicators import compute_all
from utils.candlestick import detect_all, get_latest_patterns
from utils.fundamental import full_analysis
from utils.smc_engine import full_smc_analysis
from models.ml_model import run_ml_pipeline

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title = APP_TITLE,
    page_icon  = APP_ICON,
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .buy-signal {
        background: linear-gradient(135deg, #0d2b1a, #1a4a2e);
        border: 1px solid #2ecc71;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #2ecc71;
    }
    .sell-signal {
        background: linear-gradient(135deg, #2b0d0d, #4a1a1a);
        border: 1px solid #e74c3c;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #e74c3c;
    }
    .hold-signal {
        background: linear-gradient(135deg, #2b2b0d, #4a4a1a);
        border: 1px solid #f39c12;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #f39c12;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        color: #7c83fd;
        border-bottom: 2px solid #7c83fd;
        padding-bottom: 6px;
        margin: 20px 0 12px 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 22px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/stock-market.png",
        width=80
    )
    st.sidebar.title("📈 AI Stock Screener")
    st.sidebar.markdown("*Built with ML + DL + SMC*")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Home",
         "🔍 Stock Screener",
         "📊 Deep Analysis",
         "🤖 AI Prediction",
         "📚 About"],
        label_visibility="collapsed"
    )

    st.sidebar.divider()
    ticker = st.sidebar.selectbox(
        "Select Stock",
        TICKERS,
        index=0,
    )

    period = st.sidebar.selectbox(
        "Time Period",
        ["6mo", "1y", "2y", "3y", "5y"],
        index=2,
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Built by** [Khan Atif](https://github.com/Khan7Atif9)"
    )

    return page, ticker, period


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════

def render_home():
    st.title("📈 AI Stock Screener")
    st.markdown(
        "##### ML + Deep Learning + Smart Money Concepts for NSE/BSE"
    )
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Stocks Covered", len(TICKERS))
    with col2:
        st.metric("🤖 ML Models", "RF + XGBoost")
    with col3:
        st.metric("🧠 DL Models", "LSTM + GRU")
    with col4:
        st.metric("📐 Indicators", "30+")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 What this app does")
        st.markdown("""
- **Fundamental Analysis** — 10-point scorecard from your notes
- **Technical Analysis** — 30+ indicators (RSI, MACD, BB, ATR...)
- **Candlestick Patterns** — 13 patterns detected automatically
- **SMC Signals** — BOS, CHoCH, Order Blocks, FVG
- **ML Signal** — Random Forest + XGBoost ensemble
- **DL Forecast** — LSTM price prediction (30 days)
- **Valuation** — P/E, PEG, P/B, DCF methods
        """)

    with col2:
        st.markdown("### 📐 Analysis Framework")
        st.markdown("""
**Stage 1 — Gatekeeping (10 pts)**
Sales Growth · Profit Growth · OPM
ROE · ROCE · D/E · Current Ratio
Interest Coverage · P/E · ROA

**Stage 2 — Deep Trend**
Quarterly Results · P&L Trend
Balance Sheet · Cash Flows
Shareholding Pattern

**Technical + SMC Layer**
BOS · CHoCH · Order Blocks · FVG
30+ Indicators · 13 Candle Patterns
        """)

    st.divider()
    st.markdown("### 🚀 Quick Start")
    st.info(
        "👈 Select a stock from the sidebar → "
        "Go to **Stock Screener** or **Deep Analysis**"
    )


# ═══════════════════════════════════════════════════════════════
# CANDLESTICK CHART
# ═══════════════════════════════════════════════════════════════

def plot_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Full candlestick chart with volume + indicators."""

    df_ind = compute_all(df)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
    )

    # ── Candlestick ───────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x     = df.index,
        open  = df["open"],
        high  = df["high"],
        low   = df["low"],
        close = df["close"],
        name  = ticker,
        increasing_line_color = "#2ecc71",
        decreasing_line_color = "#e74c3c",
    ), row=1, col=1)

    # ── EMAs ──────────────────────────────────────────────────
    colors = {"ema_21": "#f39c12", "ema_50": "#3498db", "ema_200": "#9b59b6"}
    for ema, color in colors.items():
        if ema in df_ind.columns:
            fig.add_trace(go.Scatter(
                x=df_ind.index, y=df_ind[ema],
                name=ema.upper(), line=dict(color=color, width=1),
                opacity=0.8,
            ), row=1, col=1)

    # ── Volume ────────────────────────────────────────────────
    colors_vol = [
        "#2ecc71" if c >= o else "#e74c3c"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"],
        name="Volume", marker_color=colors_vol,
        opacity=0.7,
    ), row=2, col=1)

    # ── RSI ───────────────────────────────────────────────────
    if "rsi" in df_ind.columns:
        fig.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind["rsi"],
            name="RSI", line=dict(color="#e67e22", width=1.5),
        ), row=3, col=1)

        fig.add_hline(y=70, line_dash="dash",
                      line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash",
                      line_color="green", opacity=0.5, row=3, col=1)

    # ── Layout ────────────────────────────────────────────────
    fig.update_layout(
        title          = f"{ticker} — Price Chart",
        template       = "plotly_dark",
        height         = 650,
        xaxis_rangeslider_visible = False,
        showlegend     = True,
        legend         = dict(
            orientation="h", y=1.02, x=0
        ),
        margin         = dict(l=0, r=0, t=50, b=0),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)

    return fig


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — STOCK SCREENER
# ═══════════════════════════════════════════════════════════════

def render_screener(ticker: str, period: str):
    st.title("🔍 Stock Screener")
    st.markdown(f"#### Analyzing **{ticker}**")
    st.divider()

    # Load data
    with st.spinner(f"Fetching data for {ticker}..."):
        df   = get_ohlcv(ticker, period)
        fund = get_fundamentals(ticker)

    if df.empty:
        st.error(f"Could not fetch data for {ticker}")
        return

    # ── Chart ─────────────────────────────────────────────────
    st.plotly_chart(
        plot_candlestick(df, ticker),
        use_container_width=True
    )

    # ── Key Metrics Row ───────────────────────────────────────
    st.markdown(
        '<p class="section-header">📊 Key Metrics</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    price = fund.get("current_price", 0)
    c1.metric("Price",    f"₹{price:,.1f}" if price else "N/A")
    c2.metric("P/E",      f"{fund.get('pe_ratio', 'N/A')}")
    c3.metric("ROE",      f"{fund.get('roe', 'N/A')}%")
    c4.metric("D/E",      f"{fund.get('debt_to_equity', 'N/A')}")
    c5.metric("OPM",      f"{fund.get('operating_margins', 'N/A')}%")
    c6.metric("Mkt Cap",  _format_mcap(fund.get("market_cap", 0)))

    # ── Fundamental Score ─────────────────────────────────────
    st.markdown(
        '<p class="section-header">📋 Fundamental Scorecard</p>',
        unsafe_allow_html=True
    )

    with st.spinner("Running fundamental analysis..."):
        analysis = full_analysis(fund)

    s1 = analysis["stage1"]

    col1, col2 = st.columns([1, 2])

    with col1:
        score = s1["total"]
        color = "#2ecc71" if score >= 7 else "#f39c12" if score >= 6.5 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color:{color}; font-size:48px; margin:0">
                {score}
            </h2>
            <p style="color:#aaa; margin:4px 0">out of 10</p>
            <p style="color:{color}; font-weight:bold">
                {s1['verdict']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Valuation")
        val = analysis["valuation"]
        if "pe_valuation" in val:
            pv = val["pe_valuation"]
            st.markdown(f"**Fair Price:** ₹{pv['fair_price']}")
            st.markdown(f"**Buy Below:** ₹{pv['buy_below']}")
            st.markdown(f"**Verdict:** {pv['verdict']}")
        if "peg_ratio" in val:
            st.markdown(
                f"**PEG Ratio:** {val['peg_ratio']['peg']} "
                f"— {val['peg_ratio']['verdict']}"
            )

    with col2:
        st.markdown("#### Score Breakdown")
        scores  = s1["scores"]
        details = s1["details"]

        labels = {
            "sales_growth":   "Sales Growth",
            "profit_growth":  "Profit Growth",
            "opm":            "OPM %",
            "roe":            "ROE",
            "roce":           "ROCE",
            "debt_equity":    "D/E Ratio",
            "current_ratio":  "Current Ratio",
            "interest_cover": "Interest Coverage",
            "pe_ratio":       "P/E vs Industry",
            "roa":            "ROA",
        }

        for key, label in labels.items():
            score_val = scores.get(key, 0)
            detail    = details.get(key, "")
            bar_pct   = int(score_val * 100)
            col_a, col_b, col_c = st.columns([2, 3, 2])
            col_a.markdown(f"**{label}**")
            col_b.progress(bar_pct)
            col_c.markdown(f"`{score_val}/1`")

    # ── Final Verdict ─────────────────────────────────────────
    st.divider()
    verdict = analysis["final_verdict"]
    if "BUY" in verdict:
        st.markdown(
            f'<div class="buy-signal">{verdict}</div>',
            unsafe_allow_html=True
        )
    elif "AVOID" in verdict:
        st.markdown(
            f'<div class="sell-signal">{verdict}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="hold-signal">{verdict}</div>',
            unsafe_allow_html=True
        )

    # ── Candlestick Patterns ──────────────────────────────────
    st.markdown(
        '<p class="section-header">🕯️ Candlestick Patterns</p>',
        unsafe_allow_html=True
    )

    with st.spinner("Detecting patterns..."):
        patterns = get_latest_patterns(df)

    bull_p = [k for k, v in patterns.items()
              if v == 1 and "bear" not in k and "score" not in k]
    bear_p = [k for k, v in patterns.items()
              if v == 1 and "bear" in k and "score" not in k]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Bullish Patterns", len(bull_p))
        for p in bull_p:
            st.success(f"✅ {p.replace('_', ' ').title()}")
    with col2:
        st.metric("Bearish Patterns", len(bear_p))
        for p in bear_p:
            st.error(f"🔴 {p.replace('_', ' ').title()}")
    with col3:
        st.metric("Bullish Score", patterns.get("bullish_score", 0))
        st.metric("Bearish Score", patterns.get("bearish_score", 0))


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — DEEP ANALYSIS
# ═══════════════════════════════════════════════════════════════

def render_analysis(ticker: str, period: str):
    st.title("📊 Deep Analysis")
    st.markdown(f"#### SMC + Technical Analysis — **{ticker}**")
    st.divider()

    with st.spinner("Loading data..."):
        df   = get_ohlcv(ticker, period)
        fund = get_fundamentals(ticker)

    if df.empty:
        st.error("No data available")
        return

    # ── SMC Analysis ──────────────────────────────────────────
    st.markdown(
        '<p class="section-header">🏦 Smart Money Concepts</p>',
        unsafe_allow_html=True
    )

    with st.spinner("Running SMC analysis..."):
        smc = full_smc_analysis(df)

    summary = smc["summary"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SMC Bias",      summary["bias"])
    c2.metric("Bull Score",    summary["bull_score"])
    c3.metric("Bear Score",    summary["bear_score"])
    c4.metric("Active OBs",
              f"🟢{summary['active_bull_obs']} 🔴{summary['active_bear_obs']}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bullish BOS",   summary["recent_bull_bos"])
    col2.metric("Bearish BOS",   summary["recent_bear_bos"])
    col3.metric("Bull CHoCH",    summary["recent_bull_choch"])
    col4.metric("Bear CHoCH",    summary["recent_bear_choch"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Active FVGs",   summary["active_fvgs"])
    col2.metric("Recent Sweeps", summary["recent_sweeps"])
    col3.metric("Price in OB",
                "✅ YES" if summary["price_in_bull_ob"] else "❌ NO")

    # ── Technical Indicators ──────────────────────────────────
    st.markdown(
        '<p class="section-header">📐 Technical Indicators</p>',
        unsafe_allow_html=True
    )

    df_ind = compute_all(df)
    last   = df_ind.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RSI (14)",
                f"{last.get('rsi', 0):.1f}",
                delta="Overbought" if last.get("rsi_ob") else
                      "Oversold"   if last.get("rsi_os") else "Normal")
    col2.metric("MACD Signal",
                "🟢 Bullish" if last.get("macd_bullish") else "🔴 Bearish")
    col3.metric("BB Position",
                f"{last.get('bb_position', 0):.2f}")
    col4.metric("ATR %",
                f"{last.get('atr_pct', 0):.2f}%")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Golden Cross",
                "✅ Yes" if last.get("golden_cross") else "❌ No")
    col2.metric("Above 200 SMA",
                "✅ Yes" if last.get("above_200sma") else "❌ No")
    col3.metric("EMA Aligned",
                "✅ Yes" if last.get("ema_aligned") else "❌ No")
    col4.metric("Volatility",
                f"{last.get('volatility', 0):.1f}%")

    # ── SMC Chart with Order Blocks ───────────────────────────
    st.markdown(
        '<p class="section-header">📈 SMC Chart</p>',
        unsafe_allow_html=True
    )
    st.plotly_chart(
        plot_smc_chart(df, smc, ticker),
        use_container_width=True
    )


def plot_smc_chart(df, smc, ticker):
    """Candlestick chart with SMC levels drawn."""
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name=ticker,
        increasing_line_color="#2ecc71",
        decreasing_line_color="#e74c3c",
    ))

    # Active Order Blocks as rectangles
    for ob in smc["order_blocks"]:
        if ob["active"]:
            color = "rgba(46,204,113,0.15)" if ob["type"] == "Bullish OB" \
                    else "rgba(231,76,60,0.15)"
            border = "#2ecc71" if ob["type"] == "Bullish OB" else "#e74c3c"
            fig.add_hrect(
                y0=ob["low"], y1=ob["high"],
                fillcolor=color,
                line=dict(color=border, width=1),
                annotation_text=ob["type"],
                annotation_position="right",
            )

    # FVGs as horizontal bands
    for fvg in smc["fvgs"]:
        if fvg["active"]:
            color = "rgba(52,152,219,0.1)"
            fig.add_hrect(
                y0=fvg["low"], y1=fvg["high"],
                fillcolor=color,
                line=dict(color="#3498db", width=0.5, dash="dot"),
                annotation_text=fvg["type"],
                annotation_position="right",
            )

    fig.update_layout(
        title    = f"{ticker} — SMC Analysis",
        template = "plotly_dark",
        height   = 550,
        xaxis_rangeslider_visible=False,
        margin   = dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — AI PREDICTION
# ═══════════════════════════════════════════════════════════════

def render_prediction(ticker: str, period: str):
    st.title("🤖 AI Prediction")
    st.markdown(f"#### ML Signal + DL Forecast — **{ticker}**")
    st.divider()

    with st.spinner("Loading data..."):
        df = get_ohlcv(ticker, "3y")   # Need 3y for training

    if df.empty:
        st.error("No data available")
        return

    # ── ML Signal ─────────────────────────────────────────────
    st.markdown(
        '<p class="section-header">🤖 ML Signal (RF + XGBoost)</p>',
        unsafe_allow_html=True
    )

    with st.spinner("Running ML models..."):
        ml = run_ml_pipeline(ticker, df)

    if "error" in ml:
        st.error(ml["error"])
    else:
        signal = ml.get("signal", "HOLD")
        conf   = ml.get("confidence", 0)
        css    = ("buy-signal"  if signal == "BUY"  else
                  "sell-signal" if signal == "SELL" else
                  "hold-signal")

        st.markdown(
            f'<div class="{css}">🤖 ML Signal: {signal} '
            f'— Confidence: {conf}%</div>',
            unsafe_allow_html=True
        )
        st.markdown("")

        col1, col2, col3, col4 = st.columns(4)
        proba = ml.get("probabilities", {})
        col1.metric("BUY Probability",  f"{proba.get('BUY', 0)}%")
        col2.metric("HOLD Probability", f"{proba.get('HOLD', 0)}%")
        col3.metric("SELL Probability", f"{proba.get('SELL', 0)}%")
        col4.metric("Models Agree",
                    "✅ Yes" if ml.get("agreement") else "⚠️ No")

        col1, col2 = st.columns(2)
        col1.metric("Random Forest",  ml.get("rf_signal", "N/A"))
        col2.metric("XGBoost",        ml.get("xgb_signal", "N/A"))

        # Probability Bar Chart
        fig_prob = go.Figure(go.Bar(
            x     = ["SELL", "HOLD", "BUY"],
            y     = [
                proba.get("SELL", 0),
                proba.get("HOLD", 0),
                proba.get("BUY",  0),
            ],
            marker_color = ["#e74c3c", "#f39c12", "#2ecc71"],
            text  = [
                f"{proba.get('SELL',0)}%",
                f"{proba.get('HOLD',0)}%",
                f"{proba.get('BUY', 0)}%",
            ],
            textposition = "auto",
        ))
        fig_prob.update_layout(
            title    = "Signal Probabilities",
            template = "plotly_dark",
            height   = 300,
            margin   = dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    # ── DL Forecast ───────────────────────────────────────────
    st.markdown(
        '<p class="section-header">🧠 LSTM Price Forecast (30 Days)</p>',
        unsafe_allow_html=True
    )

    if st.button("🚀 Run LSTM Forecast", type="primary"):
        with st.spinner("Training LSTM... this takes 2-3 minutes..."):
            from models.dl_model import DLStockModel
            model   = DLStockModel(ticker, "lstm")
            if not model._load():
                metrics = model.train(df, epochs=50)
                st.success(
                    f"✅ Trained! Accuracy: {metrics.get('accuracy_pct',0)}%"
                )
            forecast = model.forecast(df, days=30)

        if "error" in forecast:
            st.error(forecast["error"])
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Current Price",
                f"₹{forecast['current_price']:,.1f}"
            )
            col2.metric(
                "30-Day Forecast",
                f"₹{forecast['final_price']:,.1f}",
                delta=f"{forecast['change_pct']}%"
            )
            col3.metric("Direction", forecast["direction"])

            # Forecast Chart
            fig_fc = go.Figure()

            # Historical (last 60 days)
            hist = df["close"].iloc[-60:]
            fig_fc.add_trace(go.Scatter(
                x    = hist.index,
                y    = hist.values,
                name = "Historical",
                line = dict(color="#3498db", width=2),
            ))

            # Forecast
            fig_fc.add_trace(go.Scatter(
                x    = forecast["forecast_dates"],
                y    = forecast["forecast_prices"],
                name = "LSTM Forecast",
                line = dict(
                    color="#2ecc71", width=2, dash="dash"
                ),
                fill = "tonexty" if False else None,
            ))

            fig_fc.update_layout(
                title    = f"{ticker} — 30-Day LSTM Forecast",
                template = "plotly_dark",
                height   = 400,
                margin   = dict(l=0, r=0, t=40, b=0),
                xaxis_title = "Date",
                yaxis_title = "Price (₹)",
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            st.info(
                f"🎯 Model Accuracy: {forecast['accuracy_pct']}% | "
                f"This is a prediction, not financial advice."
            )


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═══════════════════════════════════════════════════════════════

def render_about():
    st.title("📚 About This Project")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### 🧠 Tech Stack
- **Python 3.11**
- **Streamlit** — Web UI
- **yfinance** — Stock data
- **Scikit-learn** — Random Forest
- **XGBoost** — Gradient Boosting
- **TensorFlow/Keras** — LSTM + GRU
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing

### 📊 Models Used
| Model | Purpose |
|-------|---------|
| Random Forest | Buy/Sell/Hold signal |
| XGBoost | Price direction |
| LSTM | 30-day forecast |
| GRU | 30-day forecast |
| Ensemble | Combined signal |
        """)

    with col2:
        st.markdown("""""")

    st.divider()
    st.markdown("""
    **Built by Khan Atif** |
    [GitHub](https://github.com/Khan7Atif9) |
    
    """)


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _format_mcap(val):
    if not val or val == 0:
        return "N/A"
    if val >= 1e12:
        return f"₹{val/1e12:.1f}T"
    if val >= 1e9:
        return f"₹{val/1e9:.1f}B"
    if val >= 1e7:
        return f"₹{val/1e7:.1f}Cr"
    return f"₹{val:,.0f}"


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    page, ticker, period = render_sidebar()

    if page == "🏠 Home":
        render_home()
    elif page == "🔍 Stock Screener":
        render_screener(ticker, period)
    elif page == "📊 Deep Analysis":
        render_analysis(ticker, period)
    elif page == "🤖 AI Prediction":
        render_prediction(ticker, period)
    elif page == "📚 About":
        render_about()


if __name__ == "__main__":
    main()
### 📐 Analysis Methods
##- **Fundamental Analysis** — Module 3 scoring
##- **Technical Analysis** — 30+ indicators
##- **Candlestick Patterns** — 13 patterns
##- **SMC** — BOS, CHoCH, OB, FVG
##- **Valuation** — P/E, PEG, P/B, DCF

### 🏗️ Project Structure