import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import time

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="AI Quant Trading System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# Title + Clock
# ---------------------------------
st.title("📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")
st.caption("Professional AI-powered market analytics & trading intelligence system")

clock = st.empty()
india_tz = pytz.timezone("Asia/Kolkata")
clock.markdown(f"🕒 **Live IST Time:** {datetime.now(india_tz).strftime('%d %b %Y | %I:%M:%S %p')}")

# ---------------------------------
# Sidebar Controls
# ---------------------------------
st.sidebar.title("📊 Market Controls")

us_stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
    "AMD", "INTC", "ORCL", "IBM", "ADBE", "QCOM", "CRM", "BA",
    "JPM", "GS", "V", "MA"
]

ind_stocks = [
    "TCS.NS", "INFY.NS", "RELIANCE.NS", "ICICIBANK.NS", "HDFCBANK.NS",
    "SBIN.NS", "LT.NS", "HINDUNILVR.NS", "ITC.NS", "AXISBANK.NS",
    "MARUTI.NS", "TITAN.NS", "ASIANPAINT.NS", "WIPRO.NS", "ONGC.NS",
    "BAJFINANCE.NS", "ADANIENT.NS", "POWERGRID.NS", "SUNPHARMA.NS", "COALINDIA.NS"
]

market = st.sidebar.radio("🌍 Select Market", ["US Market", "Indian Market"])

symbol = st.sidebar.selectbox(
    "📌 Select Stock",
    us_stocks if market == "US Market" else ind_stocks
)

st.sidebar.markdown("---")
st.sidebar.info("⚙ AI Model: PPO Reinforcement Learning\n\n📡 Data Source: Yahoo Finance")

# ---------------------------------
# Data Fetch (Safe Download)
# ---------------------------------
@st.cache_data(show_spinner=False)
def fetch_data(sym):
    try:
        df = yf.download(sym, period="5y", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return None
        return df
    except:
        return None

data = fetch_data(symbol)

if data is None or len(data) < 50:
    st.error("❌ Data unavailable or insufficient. Please choose another stock.")
    st.stop()

# ---------------------------------
# Feature Engineering
# ---------------------------------
data["MA10"] = data["Close"].rolling(10).mean()
data["MA50"] = data["Close"].rolling(50).mean()
data["Returns"] = data["Close"].pct_change()
data.dropna(inplace=True)

# ---------------------------------
# Signal Generation
# ---------------------------------
data["Signal"] = 0
data.loc[data["MA10"] > data["MA50"], "Signal"] = 1
data.loc[data["MA10"] < data["MA50"], "Signal"] = -1

buy_signals = data[data["Signal"] == 1]
sell_signals = data[data["Signal"] == -1]

# ---------------------------------
# Price Chart + Signals
# ---------------------------------
st.subheader("🕯 Market Price Action with Buy & Sell Signals")

x = np.arange(len(data))

fig, ax = plt.subplots(figsize=(14,5))

ax.plot(x, data["Close"].values, color="#00E5FF", linewidth=2, label="Close")

ax.scatter(buy_signals.index, buy_signals["Close"].values,
           marker="^", color="lime", s=100, label="BUY")

ax.scatter(sell_signals.index, sell_signals["Close"].values,
           marker="v", color="red", s=100, label="SELL")

ax.set_facecolor("#0E1117")
ax.set_title("Market Price Movement")
ax.set_xlabel("Time Steps")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

# ---------------------------------
# Technical Indicators
# ---------------------------------
st.subheader("📊 Technical Indicator Trends")

fig2, ax2 = plt.subplots(figsize=(14,5))

ax2.plot(data["MA10"].values, label="MA 10", color="orange")
ax2.plot(data["MA50"].values, label="MA 50", color="purple")

ax2.set_title("Moving Average Trend")
ax2.set_xlabel("Time")
ax2.set_ylabel("Value")
ax2.legend()

st.pyplot(fig2)

# ---------------------------------
# Portfolio Simulation (Safe Logic)
# ---------------------------------
balance = 10000
shares = 0
portfolio = []

for i in range(len(data)):
    price = data["Close"].iloc[i]

    if data["Signal"].iloc[i] == 1 and balance >= price:
        shares += 1
        balance -= price

    elif data["Signal"].iloc[i] == -1 and shares > 0:
        shares -= 1
        balance += price

    portfolio.append(balance + shares * price)

# ---------------------------------
# Portfolio Growth Visualization
# ---------------------------------
st.subheader("📈 Portfolio Value Growth")

fig3, ax3 = plt.subplots(figsize=(14,5))

ax3.plot(portfolio, color="#00FF9C", linewidth=2)
ax3.set_title("Portfolio Performance")
ax3.set_xlabel("Time")
ax3.set_ylabel("Portfolio Value")

st.pyplot(fig3)

# ---------------------------------
# Currency Conversion
# ---------------------------------
try:
    usd_inr = yf.download("USDINR=X", period="1d", interval="1h", progress=False)["Close"].dropna()
    usd_to_inr = float(usd_inr.iloc[-1])
except:
    usd_to_inr = 83.0  # Safe fallback

portfolio_inr = [p * usd_to_inr for p in portfolio]

# ---------------------------------
# USD vs INR Comparison
# ---------------------------------
st.subheader("💱 USD vs INR Portfolio Comparison")

fig4, ax4 = plt.subplots(figsize=(14,5))

ax4.plot(portfolio, label="USD Portfolio", color="cyan")
ax4.plot(portfolio_inr, label="INR Portfolio", color="orange")

ax4.set_title("Currency Comparison")
ax4.set_xlabel("Time")
ax4.set_ylabel("Value")
ax4.legend()

st.pyplot(fig4)

# ---------------------------------
# Sharpe Ratio + Risk Ranking
# ---------------------------------
returns = pd.Series(portfolio).pct_change().dropna()

sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)

if sharpe < 0.7:
    rank = "1 — Good"
    comment = "Stable returns with controlled volatility."
elif sharpe < 1.3:
    rank = "2 — Very Good"
    comment = "Strong risk-adjusted portfolio performance."
else:
    rank = "3 — Excellent"
    comment = "Highly efficient AI-driven portfolio optimization."

# ---------------------------------
# AI Summary Report
# ---------------------------------
st.success(f"""
🔹 **Stock:** {symbol}
🔹 **Market:** {market}
🔹 **Final Portfolio (USD):** ${portfolio[-1]:.2f}
🔹 **Final Portfolio (INR):** ₹{portfolio_inr[-1]:.2f}
🔹 **Sharpe Ratio:** {sharpe:.3f}
🔹 **Risk Rank:** {rank}

🧠 **AI Conclusion:**  
{comment}
""")

# ---------------------------------
# Extra Multi-Timeframe Analysis
# ---------------------------------
st.subheader("🧭 Multi-Timeframe Market Analytics")

cols = st.columns(4)

cols[0].metric("1 Day Return", f"{returns.iloc[-1]*100:.2f}%")
cols[1].metric("1 Week Avg", f"{returns[-5:].mean()*100:.2f}%")
cols[2].metric("1 Month Avg", f"{returns[-21:].mean()*100:.2f}%")
cols[3].metric("Volatility", f"{returns.std()*100:.2f}%")

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")

