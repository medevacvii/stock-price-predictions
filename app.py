import streamlit as st
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# =====================================================
# Configuration
# =====================================================
DATA_TTL_SECONDS = 30
ROLLING_WINDOW = 15
MAX_SYMBOLS = 5

NYSE_TZ = pytz.timezone("US/Eastern")
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)

# =====================================================
# Session State
# =====================================================
if "symbols" not in st.session_state:
    st.session_state.symbols = []

# =====================================================
# Helper Functions
# =====================================================
def is_market_open(now_et: datetime) -> bool:
    return MARKET_OPEN <= now_et.time() <= MARKET_CLOSE


def minutes_until_close(now_et: datetime) -> int:
    close_dt = now_et.replace(
        hour=MARKET_CLOSE.hour,
        minute=MARKET_CLOSE.minute,
        second=0,
        microsecond=0
    )
    return max(int((close_dt - now_et).total_seconds() // 60), 0)


@st.cache_data(ttl=DATA_TTL_SECONDS)
def load_intraday_data(symbol: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period="7d",
        interval="1m",
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return df

    df = df.reset_index()
    df.rename(columns={"Datetime": "timestamp"}, inplace=True)

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(NYSE_TZ)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(NYSE_TZ)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    last_trading_date = df["timestamp"].dt.date.max()
    df = df[df["timestamp"].dt.date == last_trading_date]

    return df


def project_prices(df: pd.DataFrame, minutes_forward: int) -> pd.DataFrame:
    recent = df.tail(ROLLING_WINDOW)
    if len(recent) < 2:
        return pd.DataFrame()

    prices = recent["Close"].values
    times = np.arange(len(prices))

    slope = np.polyfit(times, prices, 1)[0]
    last_price = prices[-1]

    future_times = np.arange(1, minutes_forward + 1)
    projected_prices = last_price + slope * future_times

    future_timestamps = [
        df["timestamp"].iloc[-1] + pd.Timedelta(minutes=i)
        for i in future_times
    ]

    return pd.DataFrame({
        "timestamp": future_timestamps,
        "Close": projected_prices
    })


def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# =====================================================
# Streamlit App
# =====================================================
st.set_page_config(page_title="Intraday Stock Projection", layout="wide")

st.title("Intraday Stock Price Projection (Demo)")
st.caption(
    "Short-horizon intraday projection using live Yahoo Finance data. "
    "This is not financial advice."
)

# -----------------------------------------------------
# Add Ticker Input (Search Bar)
# -----------------------------------------------------
new_symbol = st.text_input(
    "Add a stock ticker (up to 5)",
    placeholder="e.g. TSLA, AAPL, NVDA"
).upper().strip()

col_add, col_clear = st.columns([1, 1])

with col_add:
    if st.button("Add ticker"):
        if not new_symbol:
            st.warning("Enter a ticker symbol.")
        elif not re.fullmatch(r"[A-Z.\-]{1,10}", new_symbol):
            st.error("Invalid ticker format.")
        elif new_symbol in st.session_state.symbols:
            st.warning("Ticker already added.")
        elif len(st.session_state.symbols) >= MAX_SYMBOLS:
            st.warning("You can compare up to 5 stocks.")
        else:
            st.session_state.symbols.append(new_symbol)
            safe_rerun()

with col_clear:
    if st.button("Clear all"):
        st.session_state.symbols = []
        safe_rerun()

# -----------------------------------------------------
# Selected Symbols
# -----------------------------------------------------
if not st.session_state.symbols:
    st.info("Add at least one stock to begin comparison.")
    st.stop()

st.markdown("### Selected stocks")

for sym in st.session_state.symbols.copy():
    col_sym, col_remove = st.columns([4, 1])

    with col_sym:
        st.write(sym)

    with col_remove:
        if st.button("❌", key=f"remove_{sym}"):
            st.session_state.symbols.remove(sym)
            safe_rerun()

# =====================================================
# Load Data
# =====================================================
data = {}

for sym in st.session_state.symbols:
    try:
        df = load_intraday_data(sym)

        if df.empty:
            st.warning(f"No data found for {sym}. Invalid or unsupported ticker.")
            continue

        data[sym] = df

    except Exception as e:
        st.warning(f"Failed to load {sym}")

if not data:
    st.error("No valid stock data loaded.")
    st.stop()

now_et = datetime.now(NYSE_TZ)

# =====================================================
# Plot Setup
# =====================================================
for sym, df in data.items():

    # Anchor session boundaries
    session_start = df["timestamp"].iloc[0].replace(
        hour=9, minute=30, second=0
    )
    session_end = df["timestamp"].iloc[0].replace(
        hour=16, minute=0, second=0
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.65, 0.35],
        subplot_titles=[
            f"{sym} — Intraday Price ({session_start:%b %d, %Y})",
            "Volume"
        ]
    )

    # ---- Price line ----
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["Close"],
            mode="lines",
            name="Price",
            line=dict(width=2)
        ),
        row=1,
        col=1
    )

    # ---- Projection (if market open) ----
    if is_market_open(now_et):
        mins_left = minutes_until_close(now_et)
        proj = project_prices(df, mins_left)

        if not proj.empty:
            fig.add_trace(
                go.Scatter(
                    x=proj["timestamp"],
                    y=proj["Close"],
                    mode="lines",
                    name="Projection",
                    line=dict(dash="dash", color="orange")
                ),
                row=1,
                col=1
            )

    # ---- Volume bars ----
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["Volume"],
            name="Volume",
            marker_color="rgba(150,150,150,0.4)"
        ),
        row=2,
        col=1
    )

    # ---- Session end marker ----
    session_end_ts = df["timestamp"].iloc[-1].to_pydatetime()

    fig.add_vline(
        x=session_end_ts,
        line_dash="dot",
        line_color="gray",
        row=1,
        col=1
    )

    fig.update_layout(
        height=650,
        hovermode="x unified",
        showlegend=True,
        xaxis=dict(
            range=[session_start, session_end],
            tickformat="%H:%M<br>%b %d, %Y"
        ),
        yaxis_title="Price ($)",
        yaxis2_title="Volume",
        margin=dict(t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# Footer
# =====================================================
st.caption(
    "Projection is a simple extrapolation based on recent intraday trends. "
    "Accuracy decreases rapidly with horizon."
)

# =====================================================
# Auto-refresh (market hours only)
# =====================================================
if is_market_open(now_et):
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

    if time.time() - st.session_state.last_refresh >= 5:
        st.session_state.last_refresh = time.time()
        safe_rerun()
