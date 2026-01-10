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
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)

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
        period="7d",        # wider window
        interval="1m",
        progress=False,
        auto_adjust=False
    )

    df = df.reset_index()
    df.rename(columns={"Datetime": "timestamp"}, inplace=True)

    # Normalize timezone
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(NYSE_TZ)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(NYSE_TZ)

    # 🔑 CRITICAL: derive the last available trading date from data
    last_trading_date = df["timestamp"].dt.date.max()

    # Filter to that date
    df = df[df["timestamp"].dt.date == last_trading_date]

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

def get_last_trading_day(now_et: datetime) -> datetime:
    """
    Returns the most recent weekday (Mon–Fri).
    Does NOT account for market holidays yet (acceptable for v1).
    """
    last_day = now_et

    while last_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        last_day -= pd.Timedelta(days=1)

    return last_day

def safe_rerun():
    """Version-safe Streamlit rerun."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

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
trading_day = get_last_trading_day(now_et).replace(
    hour=0, minute=0, second=0, microsecond=0
)

# Load data
df = load_intraday_data(symbol)

st.write(
    "Row count:", len(df),
    "Date values:", df["timestamp"].dt.date.unique(),
    df.head(),
    df.tail()
)

session_date = df["timestamp"].dt.date.iloc[0]

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

if not is_market_open(now_et):
    st.info(
        "Market is currently closed. Showing data from the last trading day."
    )

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

session_end_ts = df["timestamp"].iloc[-1].to_pydatetime()

# Vertical line (NO annotation here)
fig.add_vline(
    x=session_end_ts,
    line_dash="dot",
    line_color="gray"
)

# Explicit annotation (safe path)
fig.add_annotation(
    x=session_end_ts,
    y=1,
    yref="paper",
    text="Session End",
    showarrow=False,
    xanchor="left",
    yanchor="bottom",
    font=dict(color="gray")
)

# Anchor session boundaries explicitly (CRITICAL)
session_start = df["timestamp"].iloc[0].replace(
    hour=9, minute=30, second=0
)
session_end = df["timestamp"].iloc[0].replace(
    hour=16, minute=0, second=0
)

fig.update_layout(
    title=f"{symbol} — Intraday Price ({session_start:%b %d, %Y})",
    xaxis_title="Time (ET)",
    yaxis_title="Price ($)",
    hovermode="x unified",
    xaxis=dict(
        range=[session_start, session_end],
        tickformat="%H:%M<br>%b %d, %Y"
    )
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.caption(
    "Projection is a simple extrapolation based on recent intraday trends. "
    "Accuracy decreases rapidly with horizon."
)

# -----------------------------
# Auto-refresh (only when market is open)
# -----------------------------
if is_market_open(now_et):
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

    if time.time() - st.session_state.last_refresh >= 5:
        st.session_state.last_refresh = time.time()
        safe_rerun()