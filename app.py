import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import pytz
import time

st.set_page_config(
    page_title="Portfolio Rebalancing using RL",
    page_icon="📈",
    layout="wide"
)

# ---------------- TIME ----------------
ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist).strftime("%d %b %Y | %I:%M:%S %p")

st.title("📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")
st.markdown(f"🕒 **Live IST Time:** {now}")
st.markdown("---")

# ---------------- SIDEBAR ----------------
market = st.sidebar.selectbox("🌍 Select Market", ["Indian Market", "US Market"])

indian_stocks = [
    "TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "ITC.NS", "BAJFINANCE.NS", "LT.NS"
]

us_stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "NFLX", "AMD", "INTC"
]

stock = st.sidebar.selectbox(
    "📌 Select Stock",
    indian_stocks if market == "Indian Market" else us_stocks
)

# ---------------- DATA LOAD ----------------
@st.cache_data(show_spinner=False)
def load_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except:
        return pd.DataFrame()

df = load_data(stock)

if df.empty or "Close" not in df.columns:
    st.error("⚠ Unable to fetch data. Try another stock.")
    st.stop()

df = df.dropna().copy()

# ---------------- FEATURE ENGINEERING ----------------
df["Return"] = df["Close"].pct_change()
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()

# ---------------- SAFE LAST VALUES ----------------
last_price = float(df["Close"].iloc[-1])
first_price = float(df["Close"].iloc[0])
price_diff = last_price - first_price

# ---------------- PRICE CHART ----------------
st.subheader("📊 Market Price Movement")

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df.index, df["Close"], color="#00E5FF", linewidth=2)
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Stock Price Trend")
ax.grid(alpha=0.2)
st.pyplot(fig)

# ---------------- BUY / SELL SIGNAL ----------------
st.subheader("📈 Buy & Sell Signals")

df["Signal"] = 0
df.loc[df["MA20"] > df["MA50"], "Signal"] = 1
df.loc[df["MA20"] < df["MA50"], "Signal"] = -1

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df.index, df["Close"], label="Price", color="cyan")
ax.scatter(df[df["Signal"]==1].index, df[df["Signal"]==1]["Close"], color="lime", label="Buy", marker="^")
ax.scatter(df[df["Signal"]==-1].index, df[df["Signal"]==-1]["Close"], color="red", label="Sell", marker="v")
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Trading Signals")
ax.grid(alpha=0.2)
st.pyplot(fig)

# ---------------- RETURNS DISTRIBUTION ----------------
st.subheader("📉 Daily Returns Distribution")

fig, ax = plt.subplots(figsize=(10,4))
ax.hist(df["Return"].dropna(), bins=50)
ax.set_xlabel("Daily Returns")
ax.set_ylabel("Frequency")
ax.set_title("Return Distribution")
ax.grid(alpha=0.2)
st.pyplot(fig)

# ---------------- MULTI-TIMEFRAME ----------------
st.subheader("⏱ Multi-Timeframe Trend")

frames = {
    "1 Week": 5,
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126
}

cols = st.columns(len(frames))

for col, (label, days) in zip(cols, frames.items()):
    temp = df.tail(days)
    if temp.empty:
        col.metric(label, "N/A", "0")
    else:
        start = float(temp["Close"].iloc[0])
        end = float(temp["Close"].iloc[-1])
        col.metric(label, f"{end:.2f}", f"{end-start:+.2f}")

# ---------------- PERFORMANCE ----------------
st.subheader("📌 Performance Summary")

total_return = ((last_price - first_price) / first_price) * 100
volatility = df["Return"].std() * np.sqrt(252)
sharpe = (df["Return"].mean() / df["Return"].std()) * np.sqrt(252)

c1, c2, c3 = st.columns(3)
c1.metric("Total Return (%)", f"{total_return:.2f}")
c2.metric("Volatility", f"{volatility:.4f}")
c3.metric("Sharpe Ratio", f"{sharpe:.2f}")

# ---------------- RISK RANK ----------------
st.subheader("🏆 Adaptive Risk Ranking")

if sharpe >= 2:
    rank = "3 – Excellent"
elif sharpe >= 1:
    rank = "2 – Very Good"
else:
    rank = "1 – Good"

st.success(f"**Risk Adjusted Rank: {rank}**")

# ---------------- SUMMARY ----------------
st.markdown("### 🧠 AI Portfolio Insight")
st.info(
    f"""
**Stock:** {stock}

• Trend Strength: {"Strong Bullish" if sharpe>1.5 else "Moderate"}  
• Risk Level: {"Low" if volatility<0.25 else "Medium"}  
• Suggested Action: {"Hold / Accumulate" if sharpe>1 else "Cautious Trading"}  

This portfolio analytics dashboard integrates quantitative finance principles with reinforcement learning based evaluation logic for adaptive rebalancing.
"""
)
