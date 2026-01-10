import streamlit as st
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
import pytz
import plotly.graph_objects as go

# -----------------------------
# Configuration
# -----------------------------
MAX_TICKERS = 1          # v1: single ticker
DATA_TTL_SECONDS = 30   # yfinance refresh interval
UI_REFRESH_MS = 5000    # UI refresh every 5s
ROLLING_WINDOW = 15     # minutes for slope/volatility

NYSE_TZ = pytz.timezone("US/Eastern")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Example symbol universe (expand later)
SYMBOLS = sorted([
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "AMD", "NFLX", "INTC"
])

# -----------------------------
# Helpers
# -----------------------------
def is_market_open(now_et: datetime) -> bool:
    return MARKET_OPEN <= now_et.time() <= MARKET_CLOSE


def minutes_until_close(now_et: datetime) -> int:
    close_dt = now_et.replace(
        hour=MARKET_CLOSE.hour,
        minute=MARKET_CLOSE.minute,
        second=0,
        microsecond=0
    )
    delta = close_dt - now_et
    return max(int(delta.total_seconds() // 60), 0)


@st.cache_data(ttl=DATA_TTL_SECONDS)
def load_intraday_data(symbol: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period="1d",
        interval="1m",
        progress=False
    )
    df = df.reset_index()
    df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    return df


def project_prices(df: pd.DataFrame, minutes_forward: int) -> pd.DataFrame:
    """
    Simple intraday projection using recent slope.
    This is intentionally lightweight and explainable.
    """
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


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Intraday Stock Projection", layout="wide")

st.title("Intraday Stock Price Projection (Demo)")
st.caption(
    "Short-horizon intraday projection using live Yahoo Finance data. "
    "This is not financial advice."
)

# Stock selector (searchable by default)
symbol = st.selectbox(
    "Select a stock symbol",
    options=SYMBOLS
)

now_et = datetime.now(NYSE_TZ)

# Load data
df = load_intraday_data(symbol)

if df.empty:
    st.warning("No intraday data available.")
    st.stop()

current_price = float(df["Close"].iloc[-1])
open_price = float(df["Close"].iloc[0])
pct_change = (current_price - open_price) / open_price * 100

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Today's Change", f"{pct_change:.2f}%")
col3.metric("Market Status", "OPEN" if is_market_open(now_et) else "CLOSED")

# Projection
projection_df = pd.DataFrame()
if is_market_open(now_et):
    mins_left = minutes_until_close(now_et)
    projection_df = project_prices(df, mins_left)

# -----------------------------
# Plot
# -----------------------------
fig = go.Figure()

# Actual prices
fig.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["Close"],
    mode="lines",
    name="Actual Price"
))

# Projection
if not projection_df.empty:
    fig.add_trace(go.Scatter(
        x=projection_df["timestamp"],
        y=projection_df["Close"],
        mode="lines",
        name="Projected Price",
        line=dict(dash="dash")
    ))

# "Now" marker
fig.add_vline(
    x=df["timestamp"].iloc[-1],
    line_dash="dot",
    line_color="gray"
)

fig.update_layout(
    title=f"{symbol} — Intraday Price & Projection",
    xaxis_title="Time (ET)",
    yaxis_title="Price ($)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.caption(
    "Projection is a simple extrapolation based on recent intraday trends. "
    "Accuracy decreases rapidly with horizon."
)

# -----------------------------
# Auto-refresh (every 5 seconds)
# -----------------------------
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh >= 5:
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()