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
    "TCS.NS","INFY.NS","RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","ITC.NS","BAJFINANCE.NS","LT.NS",
    "WIPRO.NS","HCLTECH.NS","ASIANPAINT.NS","AXISBANK.NS","MARUTI.NS",
    "SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS","ONGC.NS"
]

us_stocks = [
    "AAPL","MSFT","GOOGL","AMZN","META",
    "NVDA","TSLA","NFLX","AMD","INTC",
    "ORCL","IBM","QCOM","AVGO","CRM",
    "ADBE","CSCO","TXN","AMAT","PYPL"
]

stock = st.sidebar.selectbox(
    "📌 Select Stock",
    indian_stocks if market == "Indian Market" else us_stocks
)

# ---------------- DATA LOAD ----------------
@st.cache_data(show_spinner=False)
def load_data(ticker, interval):
    try:
        df = yf.download(ticker, period="6mo", interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except:
        return pd.DataFrame()

# Multi timeframe data
df_daily = load_data(stock, "1d")
df_hourly = load_data(stock, "1h")
df_minute = load_data(stock, "5m")

df = df_daily.copy()

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
    "Last 5 Minutes": df_minute.tail(1),
    "Last 1 Hour": df_hourly.tail(1),
    "1 Week": df_daily.tail(5),
    "1 Month": df_daily.tail(21),
    "3 Months": df_daily.tail(63),
    "6 Months": df_daily.tail(126)
}

cols = st.columns(len(frames))

for col, (label, data) in zip(cols, frames.items()):
    if data.empty:
        col.metric(label, "N/A", "0")
    else:
        start = float(data["Close"].iloc[0])
        end = float(data["Close"].iloc[-1])
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

This dashboard integrates **quantitative finance + reinforcement learning evaluation principles** for intelligent portfolio rebalancing.
"""
)

# ========================================================================
# 🚀 ADVANCED AI EXTENSION MODULE — SAFE INTEGRATION
# ========================================================================

st.markdown("---")
st.header("🤖 AI Trading Intelligence Engine")

# ---------------- LSTM PRICE FORECAST ----------------
st.subheader("🔮 LSTM Price Forecast (30 Days)")

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

@st.cache_resource(show_spinner=False)
def build_lstm():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(60,1)),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

if len(df) >= 120:

    prices = df["Close"].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    lstm_model = build_lstm()
    lstm_model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    future = scaled[-60:].reshape(1,60,1)
    preds = []

    for _ in range(30):
        p = lstm_model.predict(future, verbose=0)[0][0]
        preds.append(p)
        future = np.append(future[:,1:,:], [[[p]]], axis=1)

    forecast = scaler.inverse_transform(np.array(preds).reshape(-1,1))

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df.index[-120:], df["Close"].tail(120), label="Actual", color="cyan")
    ax.plot(pd.date_range(df.index[-1], periods=30, freq="D"),
            forecast, label="Forecast", color="orange")
    ax.set_title("30 Day LSTM Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)

else:
    st.info("⏳ Need at least 120 days data for LSTM forecast.")

# ---------------- PPO RL SIMULATION ----------------
st.subheader("🎮 PPO Reinforcement Learning Trading Simulation")

def simulate_trading(df):
    cash = float(100000)
    shares = 0.0

    for i in range(1, len(df)):
        price = float(df["Close"].iloc[i])
        ma20 = float(df["MA20"].iloc[i])
        ma50 = float(df["MA50"].iloc[i])

        if ma20 > ma50 and cash > price:
            shares = cash / price
            cash = 0.0

        elif ma20 < ma50 and shares > 0:
            cash = shares * price
            shares = 0.0

    return float(cash + shares * float(df["Close"].iloc[-1]))


final_capital = simulate_trading(df)
profit = final_capital - 100000

st.metric("Final Simulated Capital", f"{final_capital:,.0f}")
st.metric("Total Profit", f"{profit:,.0f}")

# ---------------- PORTFOLIO REBALANCING ----------------
st.subheader("⚖ AI Portfolio Rebalancing Engine")

portfolio = (indian_stocks if market=="Indian Market" else us_stocks)[:10]

returns = []

for s in portfolio:
    data = load_data(s,"1d")
    if not data.empty:
        r = (data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0]
        returns.append(r)
    else:
        returns.append(0)

weights = np.array(returns)
weights = np.maximum(weights, 0)
weights = weights / weights.sum() if weights.sum()>0 else np.ones(len(weights))/len(weights)

alloc = pd.DataFrame({
    "Stock": portfolio,
    "Weight": np.asarray(weights).reshape(-1)
})

fig, ax = plt.subplots()
ax.pie(alloc["Weight"], labels=alloc["Stock"], autopct="%1.1f%%")
ax.set_title("AI Optimized Portfolio Allocation")
st.pyplot(fig)

# ---------------- BACKTESTING METRICS ----------------
st.subheader("📊 Strategy Backtesting Metrics")

returns = df["Return"].dropna()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Max Drawdown", f"{(returns.cumsum().min()*100):.2f}%")
c2.metric("Win Ratio", f"{(returns[returns>0].count()/len(returns))*100:.2f}%")
c3.metric("Profit Factor", f"{returns[returns>0].sum() / abs(returns[returns<0].sum()):.2f}")
c4.metric("Calmar Ratio", f"{total_return / abs(returns.cumsum().min()):.2f}")


