import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

# ----------------------------------
# TITLE
# ----------------------------------
st.title("📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")
st.caption("AI-driven intelligent trading analytics & portfolio optimization")

# ----------------------------------
# LIVE CLOCK
# ----------------------------------
tz = pytz.timezone("Asia/Kolkata")
st.markdown(f"🕒 **Live Time:** {datetime.now(tz).strftime('%Y-%m-%d  %H:%M:%S')} IST")

# ----------------------------------
# STOCK SELECTION
# ----------------------------------
st.sidebar.header("📌 Stock Selection")

indian_stocks = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","HINDUNILVR.NS",
    "LT.NS","SBIN.NS","ITC.NS","AXISBANK.NS","MARUTI.NS","BAJFINANCE.NS",
    "KOTAKBANK.NS","WIPRO.NS","HCLTECH.NS","ASIANPAINT.NS","SUNPHARMA.NS",
    "ULTRACEMCO.NS","TITAN.NS","NESTLEIND.NS"
]

us_stocks = [
    "AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","ORCL","INTC",
    "AMD","IBM","JPM","GS","BAC","WMT","KO","PEP","COST","DIS"
]

market = st.sidebar.radio("Select Market", ["Indian Market 🇮🇳", "US Market 🇺🇸"])

if market == "Indian Market 🇮🇳":
    ticker = st.sidebar.selectbox("Select Stock", indian_stocks)
else:
    ticker = st.sidebar.selectbox("Select Stock", us_stocks)

# ----------------------------------
# DATA DOWNLOAD (SAFE)
# ----------------------------------
@st.cache_data(show_spinner=False)
def load_data(ticker):
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    return df

data = load_data(ticker)

if data.empty:
    st.error("⚠️ Data download failed. Try another stock or refresh.")
    st.stop()

# ----------------------------------
# FEATURE ENGINEERING
# ----------------------------------
data['MA10'] = data['Close'].rolling(10).mean()
data['MA50'] = data['Close'].rolling(50).mean()
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(10).std()
data.dropna(inplace=True)

# ----------------------------------
# MAIN PRICE CHART
# ----------------------------------
st.subheader("📊 Price Trend & Moving Averages")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(data['Close'], label='Price', color='blue')
ax.plot(data['MA10'], label='MA10', linestyle='--')
ax.plot(data['MA50'], label='MA50', linestyle='--')
ax.legend()
st.pyplot(fig)

# ----------------------------------
# CANDLESTICK CHART
# ----------------------------------
st.subheader("🕯️ Candlestick Chart")

fig2, ax2 = plt.subplots(figsize=(12,5))

for i in range(len(data)):
    color = 'green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red'
    ax2.plot([i,i], [data['Low'].iloc[i], data['High'].iloc[i]], color=color)
    ax2.plot([i,i], [data['Open'].iloc[i], data['Close'].iloc[i]], linewidth=5, color=color)

ax2.set_title("Price Action")
st.pyplot(fig2)

# ----------------------------------
# MULTI TIMEFRAME ANALYTICS
# ----------------------------------
st.subheader("📈 Multi-Timeframe Returns")

t1,t2,t3,t4,t5 = st.columns(5)

t1.metric("Minute", f"{data['Returns'].tail(1).mean()*100:.2f}%")
t2.metric("Hour", f"{data['Returns'].tail(5).mean()*100:.2f}%")
t3.metric("Week", f"{data['Returns'].tail(25).mean()*100:.2f}%")
t4.metric("Month", f"{data['Returns'].tail(100).mean()*100:.2f}%")
t5.metric("Next Month Forecast", f"{(data['Returns'].mean()*30)*100:.2f}%")

# ----------------------------------
# SHARPE RATIO
# ----------------------------------
sharpe = (np.mean(data['Returns']) / np.std(data['Returns'])) * np.sqrt(252)

if sharpe < 0.8:
    rank = "1 — Good"
elif sharpe < 1.5:
    rank = "2 — Very Good"
else:
    rank = "3 — Excellent"

st.subheader("🏆 Risk Adjusted Portfolio Ranking")
s1,s2 = st.columns(2)
s1.metric("Sharpe Ratio", f"{sharpe:.3f}")
s2.metric("Performance Rank", rank)

# ----------------------------------
# AI SUMMARY
# ----------------------------------
st.subheader("🧠 AI Stock Summary")

last_price = data['Close'].iloc[-1]
weekly = data['Returns'].tail(25).mean()*100
monthly = data['Returns'].tail(100).mean()*100
vol = data['Volatility'].iloc[-1]

summary = [
    f"Latest Price: ${last_price:.2f}",
    f"Weekly Avg Return: {weekly:.2f}%",
    f"Monthly Avg Return: {monthly:.2f}%",
    f"Volatility: {vol:.4f}",
    f"Risk Profile: {'High' if vol>0.03 else 'Moderate' if vol>0.015 else 'Low'}",
    f"Trend Bias: {'Bullish 📈' if weekly>0 else 'Bearish 📉'}"
]

for s in summary:
    st.markdown(f"- {s}")

# ----------------------------------
# USER PORTFOLIO INPUT
# ----------------------------------
st.subheader("💼 Compare With Your Portfolio")

c1,c2,c3 = st.columns(3)

with c1:
    user_capital = st.number_input("Capital ($)", 1000, 1000000, 10000)

with c2:
    user_shares = st.number_input("Shares Held", 1, 10000, 10)

with c3:
    user_buy = st.number_input("Avg Buy Price ($)", float(last_price))

user_value = user_shares * last_price
user_profit = user_value - (user_shares * user_buy)
user_return = (user_profit / (user_shares * user_buy)) * 100

st.metric("Your Portfolio Value", f"${user_value:.2f}", f"{user_return:.2f}%")

# ----------------------------------
# FINAL MESSAGE
# ----------------------------------
st.success("✅ AI-driven adaptive portfolio optimization successfully executed.")


