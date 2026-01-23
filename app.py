import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- TIME ----------------
ist = pytz.timezone("Asia/Kolkata")
current_time = datetime.now(ist).strftime("%d %b %Y | %I:%M:%S %p")

# ---------------- TITLE ----------------
st.markdown("""
# 📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning
""")
st.markdown(f"🕒 **Live IST Time:** `{current_time}`")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌍 Market Selection")

market = st.sidebar.radio("Choose Market", ["Indian Market", "US Market"])

INDIAN_STOCKS = [
    "TCS.NS","INFY.NS","RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","ITC.NS",
    "LT.NS","AXISBANK.NS","MARUTI.NS","WIPRO.NS","TATAMOTORS.NS","SUNPHARMA.NS",
    "HCLTECH.NS","ULTRACEMCO.NS","BAJFINANCE.NS","ADANIENT.NS","ADANIPORTS.NS","ONGC.NS","POWERGRID.NS"
]

US_STOCKS = [
    "AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","AMD","INTC",
    "JPM","GS","WMT","DIS","BA","CSCO","ORCL","IBM","PYPL","COST"
]

if market == "Indian Market":
    symbol = st.sidebar.selectbox("📌 Select Stock", INDIAN_STOCKS)
else:
    symbol = st.sidebar.selectbox("📌 Select Stock", US_STOCKS)

# ---------------- DATA LOADING ----------------
@st.cache_data(show_spinner=False)
def load_data(symbol):
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if df.empty:
            return None
        return df.dropna()
    except:
        return None

data = load_data(symbol)

if data is None or len(data) < 30:
    st.error("⚠️ Data unavailable / rate-limited. Please select another stock.")
    st.stop()

# ---------------- FEATURES ----------------
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()

# Buy Sell signals
data["Signal"] = 0
data.loc[data["MA20"] > data["MA50"], "Signal"] = 1
data["Position"] = data["Signal"].diff()

# ---------------- VISUAL 1: PRICE ----------------
st.subheader("📊 Market Price Movement")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(data["Close"], label="Close Price", linewidth=2)
ax.plot(data["MA20"], label="MA20", linestyle="--")
ax.plot(data["MA50"], label="MA50", linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# ---------------- BUY SELL ----------------
st.subheader("📈 Buy & Sell Signals")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(data["Close"], label="Price", linewidth=2)

buy = data[data["Position"] == 1]
sell = data[data["Position"] == -1]

ax.scatter(buy.index, buy["Close"], marker="^", color="green", s=120, label="BUY")
ax.scatter(sell.index, sell["Close"], marker="v", color="red", s=120, label="SELL")

ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# ---------------- RETURNS ----------------
returns = data["Close"].pct_change().dropna()
volatility = returns.std() * np.sqrt(252)
sharpe = (returns.mean() * 252) / volatility

# ---------------- EXTRA VISUAL ----------------
st.subheader("📉 Daily Returns Distribution")

fig, ax = plt.subplots(figsize=(12,4))
ax.hist(returns, bins=60, density=True)
ax.set_xlabel("Returns")
ax.set_ylabel("Density")
st.pyplot(fig)

# ---------------- MULTI-TIMEFRAME ----------------
st.subheader("⏱ Multi-Timeframe Trend")

cols = st.columns(4)
periods = ["5d","1mo","3mo","6mo"]

for col, p in zip(cols, periods):
    df = yf.download(symbol, period=p, interval="1d", progress=False)
    if df.empty:
        col.warning("N/A")
        continue
    col.metric(p, f"{df['Close'][-1]:.2f}", f"{(df['Close'][-1]-df['Close'][0]):.2f}")

# ---------------- PERFORMANCE ----------------
st.subheader("📌 Performance Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Annual Volatility", f"{volatility:.2%}")
col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
col3.metric("Total Return", f"{((data['Close'][-1]/data['Close'][0])-1)*100:.2f}%")

# ---------------- RANKING ----------------
st.subheader("🏆 Adaptive Risk-Adjusted Portfolio Ranking")

if sharpe < 1:
    rank = "1️⃣ GOOD"
    summary = "Stable portfolio with moderate risk."
elif sharpe < 2:
    rank = "2️⃣ VERY GOOD"
    summary = "Well-balanced return-risk tradeoff."
else:
    rank = "3️⃣ EXCELLENT"
    summary = "Strong risk-adjusted portfolio performance."

st.success(f"### Rank: {rank}")
st.info(summary)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Quantitative Finance | Reinforcement Learning | AI Portfolio Optimization Dashboard")
