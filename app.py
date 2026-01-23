import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sklearn.preprocessing import RobustScaler
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
import pytz
import time

# -------------------------------------
# Page Config
# -------------------------------------
st.set_page_config(page_title="AI Trading Dashboard", page_icon="📈", layout="wide")

# -------------------------------------
# Live Clock (Auto Refresh)
# -------------------------------------
clock_placeholder = st.empty()
india_tz = pytz.timezone("Asia/Kolkata")

def live_clock():
    now = datetime.now(india_tz)
    clock_placeholder.markdown(f"🕒 **Current Time (IST): {now.strftime('%d %b %Y, %I:%M:%S %p')}**")

live_clock()

# -------------------------------------
# Title
# -------------------------------------
st.title("📊 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")
st.markdown("### Real-Time AI Trading Dashboard | LSTM + PPO Reinforcement Learning")

# -------------------------------------
# Load PPO Model
# -------------------------------------
ppo_model = PPO.load("ppo_trading_agent")

# -------------------------------------
# Trading Environment
# -------------------------------------
class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super().__init__()
        self.data = data.values
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = 0
        return self.data[self.current_step], {}

    def step(self, action):
        price = self.data[self.current_step][0]

        if action == 1 and self.balance >= price:
            self.shares += 1
            self.balance -= price
        elif action == 2 and self.shares > 0:
            self.shares -= 1
            self.balance += price

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        portfolio_value = self.balance + self.shares * price
        reward = portfolio_value - self.initial_balance

        return self.data[self.current_step], reward, done, False, {}

# -------------------------------------
# Indian Stock Dropdown (20+)
# -------------------------------------
indian_stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "WIPRO.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "SUNPHARMA.NS", "ONGC.NS", "NTPC.NS"
]

symbol = st.selectbox("📌 Select Indian Stock", indian_stocks)

# -------------------------------------
# Run Simulation
# -------------------------------------
if st.button("🚀 Run Trading Simulation"):

    data = yf.download(symbol, period="2y", interval="1d", progress=False)

    if data.empty or len(data) < 60:
        st.error("⚠️ Insufficient market data.")
        st.stop()

    # Features
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(10).std()
    data.dropna(inplace=True)

    features = ['Close','MA10','MA50','Returns','Volatility']
    scaler = RobustScaler()
    scaled = scaler.fit_transform(data[features])

    padded = np.zeros((scaled.shape[0], 14))
    padded[:, :5] = scaled

    env = TradingEnv(pd.DataFrame(padded))

    obs,_ = env.reset()
    obs = np.array(obs, dtype=np.float32)

    done = False
    portfolio = []

    while not done:
        obs_input = obs.reshape(1, -1)
        action,_ = ppo_model.predict(obs_input, deterministic=True)
        action = int(action.item())

        obs, reward, done, _, _ = env.step(action)
        obs = np.array(obs, dtype=np.float32)
        portfolio.append(env.balance + env.shares * obs[0])

    # -------------------------------------
    # Currency Conversion USD → INR
    # -------------------------------------
    usd_inr = yf.download("USDINR=X", period="1d", interval="1m", progress=False)["Close"][-1]

    portfolio_inr = [p * usd_inr for p in portfolio]

    # -------------------------------------
    # Dashboard Layout
    # -------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Portfolio Value (USD)")
        st.line_chart(portfolio)

    with col2:
        st.subheader("💰 Portfolio Value (INR)")
        st.line_chart(portfolio_inr)

    # -------------------------------------
    # Performance Windows
    # -------------------------------------
    st.subheader("📊 Multi-Timeframe Market Performance")

    minute = yf.download(symbol, period="1d", interval="1m", progress=False)["Close"]
    hour = yf.download(symbol, period="5d", interval="15m", progress=False)["Close"]
    week = yf.download(symbol, period="1mo", interval="1d", progress=False)["Close"]
    month = yf.download(symbol, period="6mo", interval="1d", progress=False)["Close"]

    col3, col4, col5 = st.columns(3)

    col3.line_chart(minute, height=250)
    col3.caption("Minute-Level")

    col4.line_chart(hour, height=250)
    col4.caption("Hourly")

    col5.line_chart(week, height=250)
    col5.caption("Weekly")

    st.line_chart(month, height=300)
    st.caption("Monthly Trend")

    # -------------------------------------
    # Future 1 Month Prediction (Statistical Projection)
    # -------------------------------------
    st.subheader("🔮 Next 1 Month Forecast")

    last_price = data['Close'].iloc[-1]
    trend = data['Close'].pct_change().mean()

    future_days = 22
    forecast = [last_price * (1 + trend)**i for i in range(1, future_days+1)]

    st.line_chart(forecast)

    # -------------------------------------
    # Summary
    # -------------------------------------
    st.success("✅ Trading simulation completed successfully!")

    st.metric("Final Portfolio (USD)", f"${portfolio[-1]:.2f}")
    st.metric("Final Portfolio (INR)", f"₹{portfolio_inr[-1]:,.2f}")
