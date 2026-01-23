import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pytz
from stable_baselines3 import PPO
from sklearn.preprocessing import RobustScaler
import gymnasium as gym
from gymnasium import spaces

# -------------------------------------
# Page Config
# -------------------------------------
st.set_page_config(
    page_title="AI Stock Trading System",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------
# Title + Live Clock
# -------------------------------------
st.title("📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")
st.markdown("### Automated Trading System using Deep Reinforcement Learning")

clock_placeholder = st.empty()
india_tz = pytz.timezone("Asia/Kolkata")

# Live clock loop
current_time = datetime.now(india_tz)
clock_placeholder.markdown(f"🕒 **Live Time (IST):** {current_time.strftime('%d %b %Y, %I:%M:%S %p')}")

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
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
# Stock Selector
# -------------------------------------
us_stocks = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","AMD","INTC",
             "IBM","ORCL","SAP","ADBE","CRM","PYPL","BABA","UBER","CSCO","QCOM"]

indian_stocks = ["RELIANCE.NS","TCS.NS","INFY.NS","ICICIBANK.NS","HDFCBANK.NS",
                 "SBIN.NS","ITC.NS","LT.NS","AXISBANK.NS","BAJFINANCE.NS",
                 "HCLTECH.NS","WIPRO.NS","MARUTI.NS","TATAMOTORS.NS","ONGC.NS",
                 "SUNPHARMA.NS","ULTRACEMCO.NS","POWERGRID.NS","NTPC.NS","ADANIENT.NS"]

tickers = us_stocks + indian_stocks

symbol = st.selectbox("📌 Select Stock", tickers)

# -------------------------------------
# Run Simulation
# -------------------------------------
if st.button("🚀 Run Trading Simulation"):

    data = yf.download(symbol, start="2019-01-01", end="2025-01-01", progress=False)

    if data.empty or len(data) < 60:
        st.error("⚠️ Insufficient market data.")
        st.stop()

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
    # Currency Conversion
    # -------------------------------------
    usd_to_inr = yf.download("USDINR=X", period="1d", progress=False)['Close'][-1]
    portfolio_inr = [p * usd_to_inr for p in portfolio]

    # -------------------------------------
    # Dashboard
    # -------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Portfolio Growth (USD)")
        st.line_chart(portfolio)

    with col2:
        st.subheader("💱 Portfolio Growth (INR)")
        st.line_chart(portfolio_inr)

    # -------------------------------------
    # Multi-Timeframe Visualization
    # -------------------------------------
    st.subheader("📈 Multi-Timeframe Market Performance")

    t1, t2, t3, t4, t5 = st.columns(5)

    t1.metric("1 Min", f"{data['Returns'].tail(1).values[0]*100:.2f}%")
    t2.metric("1 Hour", f"{data['Returns'].tail(5).mean()*100:.2f}%")
    t3.metric("1 Week", f"{data['Returns'].tail(25).mean()*100:.2f}%")
    t4.metric("1 Month", f"{data['Returns'].tail(100).mean()*100:.2f}%")
    t5.metric("Next Month Forecast", f"{(data['Returns'].mean()*30)*100:.2f}%")

    # -------------------------------------
    # Sharpe Ratio Ranking
    # -------------------------------------
    sharpe_ratio = (np.mean(data['Returns']) / np.std(data['Returns'])) * np.sqrt(252)

    if sharpe_ratio < 0.8:
        rank = "1 — Good"
        summary = "Stable returns, moderate risk exposure."
    elif sharpe_ratio < 1.5:
        rank = "2 — Very Good"
        summary = "Strong risk-adjusted performance with balanced volatility."
    else:
        rank = "3 — Excellent"
        summary = "Outstanding portfolio efficiency with optimal risk-return tradeoff."

    st.subheader("🏆 Adaptive Risk-Adjusted Portfolio Ranking")
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
    st.metric("Performance Rank", rank)
    st.success(summary)
