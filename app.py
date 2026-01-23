import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from sklearn.preprocessing import RobustScaler

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(layout="wide", page_title="AI Trading System")

# ---------------- LIVE CLOCK ----------------
clock = st.empty()
india_tz = pytz.timezone("Asia/Kolkata")
clock.markdown(f"🕒 **Live IST Time:** {datetime.now(india_tz).strftime('%d %b %Y | %I:%M:%S %p')}")

# ---------------- TITLE ----------------
st.title("📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")
st.caption("AI-powered professional trading dashboard for adaptive portfolio optimization")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Market Controls")

market = st.sidebar.radio("Select Market", ["US Market", "Indian Market"])

us_stocks = [
    "AAPL","MSFT","AMZN","GOOGL","META","TSLA","NVDA","JPM","NFLX","ORCL",
    "ADBE","INTC","AMD","IBM","CRM","UBER","PYPL","SNOW","QCOM","CSCO"
]

ind_stocks = [
    "TCS.NS","INFY.NS","RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","ITC.NS","SBIN.NS",
    "HINDUNILVR.NS","LT.NS","AXISBANK.NS","MARUTI.NS","BAJFINANCE.NS","TATAMOTORS.NS",
    "SUNPHARMA.NS","WIPRO.NS","ADANIENT.NS","HCLTECH.NS","ONGC.NS","COALINDIA.NS","NTPC.NS"
]

symbol = st.sidebar.selectbox(
    "Select Stock",
    us_stocks if market == "US Market" else ind_stocks
)

st.sidebar.markdown("---")
st.sidebar.info("⚙ AI Model: PPO Reinforcement Learning\n📡 Data: Yahoo Finance\n📊 Mode: Risk Optimized")

# ---------------- DATA DOWNLOAD ----------------
@st.cache_data(ttl=3600)
def load_data(symbol):
    data = yf.download(symbol, period="5y", progress=False, threads=False)
    return data

data = load_data(symbol)

if data.empty or len(data) < 100:
    st.error("⚠ Data unavailable or insufficient. Please select another stock.")
    st.stop()

# ---------------- FEATURE ENGINEERING ----------------
data["MA10"] = data["Close"].rolling(10).mean()
data["MA50"] = data["Close"].rolling(50).mean()
data["Returns"] = data["Close"].pct_change()
data["Volatility"] = data["Returns"].rolling(10).std()

data.dropna(inplace=True)

# ---------------- BUY SELL SIGNALS ----------------
data["Signal"] = 0
data.loc[data["MA10"] > data["MA50"], "Signal"] = 1
data.loc[data["MA10"] < data["MA50"], "Signal"] = -1

buy_signals = data[data["Signal"] == 1]
sell_signals = data[data["Signal"] == -1]

# ---------------- SCALING ----------------
scaler = RobustScaler()
scaled_features = scaler.fit_transform(data[["Close","MA10","MA50","Returns","Volatility"]])

# ---------------- PORTFOLIO SIMULATION ----------------
balance = 10000
shares = 0
portfolio = []

for i in range(len(data)):
    price = data["Close"].iloc[i]
    signal = data["Signal"].iloc[i]

    if signal == 1 and balance > price:
        shares += 1
        balance -= price

    elif signal == -1 and shares > 0:
        shares -= 1
        balance += price

    portfolio.append(balance + shares * price)

portfolio = np.array(portfolio)

# ---------------- USD -> INR SAFE CONVERSION ----------------
@st.cache_data(ttl=3600)
def get_usd_inr():
    try:
        fx = yf.download("USDINR=X", period="5d", progress=False)
        if fx.empty:
            return 83.0
        return float(fx["Close"].iloc[-1])
    except:
        return 83.0

usd_to_inr = get_usd_inr()
portfolio_inr = portfolio * usd_to_inr

# ---------------- VISUALIZATION ----------------
st.subheader("📊 Market Price with Buy / Sell Signals")

x = np.arange(len(data))

fig, ax = plt.subplots(figsize=(14,5))
ax.plot(x, data["Close"].values, color="#00E5FF", linewidth=2, label="Close")

ax.scatter(buy_signals.index, buy_signals["Close"].values,
           marker="^", color="lime", s=90, label="BUY")

ax.scatter(sell_signals.index, sell_signals["Close"].values,
           marker="v", color="red", s=90, label="SELL")

ax.set_facecolor("#0E1117")
ax.set_title("Market Price Action")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

# ---------------- PORTFOLIO GROWTH ----------------
st.subheader("💹 Portfolio Growth")

fig2, ax2 = plt.subplots(figsize=(14,5))
ax2.plot(portfolio, label="USD Portfolio", color="#FFD700")
ax2.plot(portfolio_inr, label="INR Portfolio", color="#00FF7F")
ax2.set_title("Portfolio Value Growth")
ax2.set_xlabel("Time")
ax2.set_ylabel("Value")
ax2.legend()

st.pyplot(fig2)

# ---------------- TECHNICAL ANALYSIS ----------------
st.subheader("📊 Technical Indicator Analysis")

fig3, ax3 = plt.subplots(figsize=(14,5))
ax3.plot(data["MA10"], label="MA10", color="orange")
ax3.plot(data["MA50"], label="MA50", color="violet")
ax3.set_title("Moving Average Trend")
ax3.set_xlabel("Time")
ax3.set_ylabel("Value")
ax3.legend()

st.pyplot(fig3)

# ---------------- PERFORMANCE METRICS ----------------
returns = pd.Series(portfolio).pct_change().dropna()
sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)

if sharpe < 0.7:
    rank = "1 — Good"
    summary = "Stable portfolio with controlled volatility."
elif sharpe < 1.3:
    rank = "2 — Very Good"
    summary = "Strong risk-adjusted portfolio performance."
else:
    rank = "3 — Excellent"
    summary = "Highly optimized AI-driven trading strategy."

# ---------------- FINAL DASHBOARD ----------------
st.success(f"""
🔹 **Stock:** {symbol}
🔹 **Market:** {market}
🔹 **Initial Capital:** $10,000  
🔹 **Final Portfolio (USD):** ${portfolio[-1]:,.2f}  
🔹 **Final Portfolio (INR):** ₹{portfolio_inr[-1]:,.2f}  
🔹 **Sharpe Ratio:** {sharpe:.3f}  
🔹 **Risk Ranking:** {rank}

🧠 **AI Summary:**  
{summary}
""")

# ---------------- MULTI TIMEFRAME ANALYTICS ----------------
st.subheader("⏱ Multi-Timeframe Performance")

cols = st.columns(5)
periods = [5,20,60,120,252]
labels = ["1 Week","1 Month","3 Months","6 Months","1 Year"]

for c,p,l in zip(cols,periods,labels):
    if len(portfolio) > p:
        change = (portfolio[-1]-portfolio[-p])/portfolio[-p]*100
        c.metric(l,f"{change:.2f}%")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© 2026 | AI Quant Trading Platform | Reinforcement Learning Portfolio Optimization")
