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
st.set_page_config(page_title="AI Quant Trading", page_icon="📊", layout="wide")

# -------------------------------------
# Dual Live Clock
# -------------------------------------
col1, col2, col3 = st.columns([3,2,2])

with col1:
    st.title("📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")

with col2:
    st.markdown(f"🕒 **IST:** {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%I:%M:%S %p')}")

with col3:
    st.markdown(f"🌍 **UTC:** {datetime.utcnow().strftime('%H:%M:%S')}")

st.markdown("---")

# -------------------------------------
# System Info Panel
# -------------------------------------
with st.expander("📘 System Overview"):
    st.markdown("""
    **Modules:**
    - Reinforcement Learning (PPO)
    - Market Simulation Environment
    - Risk Metrics + Sharpe Ratio
    - USD ↔ INR Conversion
    - Multi-Timeframe Forecast Dashboard

    **Graphs Explained:**
    - Price Curve → Stock movement
    - Portfolio Growth → RL trading profit
    - Currency Curve → INR converted returns
    """)

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
# Stock Universe
# -------------------------------------
us = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","AMD","INTC"]
fashion_us = ["NKE","LULU","UA","VFC","RL","TPR","GPS","PVH","SKX","LEVI"]

india = ["RELIANCE.NS","TCS.NS","INFY.NS","ICICIBANK.NS","HDFCBANK.NS",
         "SBIN.NS","ITC.NS","LT.NS","AXISBANK.NS","BAJFINANCE.NS"]

fashion_india = ["TRENT.NS","PAGEIND.NS","RAYMOND.NS","BATAINDIA.NS","ABFRL.NS",
                  "VMART.NS","CANTABIL.NS","METROBRAND.NS","LUXIND.NS","ADANIENT.NS"]

tickers = us + fashion_us + india + fashion_india

symbol = st.selectbox("📌 Select Stock Ticker", tickers)

# -------------------------------------
# Simulation
# -------------------------------------
if st.button("🚀 Run Trading Simulation"):

    data = yf.download(symbol, start="2019-01-01", end="2025-01-01", progress=False)

    if data.empty or len(data) < 100:
        st.error("⚠️ Insufficient data for simulation.")
        st.stop()

    # Price Chart
    st.subheader("📉 Market Price Curve")
    st.line_chart(data["Close"])

    # Feature Engineering
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(10).std()
    data.dropna(inplace=True)

    features = ['Close','MA10','MA50','Returns','Volatility']
    scaler = RobustScaler()
    scaled = scaler.fit_transform(data[features])

    padded = np.zeros((scaled.shape[0],14))
    padded[:,:5] = scaled

    env = TradingEnv(pd.DataFrame(padded))
    obs,_ = env.reset()
    obs = np.array(obs, dtype=np.float32)

    portfolio = []
    done = False

    while not done:
        obs_input = obs.reshape(1,-1)
        action,_ = ppo_model.predict(obs_input, deterministic=True)
        action = int(action.item())
        obs,reward,done,_,_ = env.step(action)
        obs = np.array(obs, dtype=np.float32)
        portfolio.append(env.balance + env.shares * obs[0])

    # USD → INR Conversion
    try:
        fx = yf.download("USDINR=X", period="5d", progress=False)
        usd_to_inr = fx["Close"].iloc[-1]
    except:
        usd_to_inr = 83.0

    portfolio_inr = [p * usd_to_inr for p in portfolio]

    colA, colB = st.columns(2)

    with colA:
        st.subheader("📊 Portfolio Growth (USD)")
        st.line_chart(portfolio)

    with colB:
        st.subheader("💱 Portfolio Growth (INR)")
        st.line_chart(portfolio_inr)

    # Multi-Timeframe Metrics
    st.subheader("⏱ Performance Dashboard")

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric("Minute", f"{data['Returns'].tail(1).mean()*100:.2f}%")
    c2.metric("Hour", f"{data['Returns'].tail(5).mean()*100:.2f}%")
    c3.metric("Week", f"{data['Returns'].tail(25).mean()*100:.2f}%")
    c4.metric("Month", f"{data['Returns'].tail(100).mean()*100:.2f}%")
    c5.metric("Next Month", f"{(data['Returns'].mean()*30)*100:.2f}%")

    # Sharpe Ratio Ranking
    sharpe = (np.mean(data['Returns']) / np.std(data['Returns'])) * np.sqrt(252)

    if sharpe < 0.8:
        rank = "1 — Good"
    elif sharpe < 1.5:
        rank = "2 — Very Good"
    else:
        rank = "3 — Excellent"

    st.subheader("🏆 Adaptive Risk Ranking")
    colS1,colS2 = st.columns(2)
    colS1.metric("Sharpe Ratio", f"{sharpe:.3f}")
    colS2.metric("Risk Rank", rank)

    st.success("✔ Trading Simulation Completed Successfully")
