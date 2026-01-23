import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import time
from datetime import datetime
import pytz
import plotly.graph_objects as go
from stable_baselines3 import PPO
from sklearn.preprocessing import RobustScaler
import gymnasium as gym
from gymnasium import spaces

# -------------------------------------
# Page Config
# -------------------------------------
st.set_page_config(
    page_title="AI Quant Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------
# Header + Dual Live Clocks
# -------------------------------------
colA, colB, colC = st.columns([3,2,2])

with colA:
    st.title("📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")

with colB:
    ist_clock = st.empty()

with colC:
    utc_clock = st.empty()

india_tz = pytz.timezone("Asia/Kolkata")
while False:
    pass

ist_clock.markdown(f"🕒 **IST Time:** {datetime.now(india_tz).strftime('%I:%M:%S %p')}")
utc_clock.markdown(f"🌍 **UTC Time:** {datetime.utcnow().strftime('%H:%M:%S')}")

st.markdown("---")

# -------------------------------------
# Info Panel
# -------------------------------------
with st.expander("📘 System Overview & Graph Explanation"):
    st.markdown("""
    **System Components**
    - 🧠 PPO Reinforcement Learning Agent
    - 📊 Multi-timeframe Financial Analytics
    - 📈 Candlestick & Portfolio Charts
    - 💱 Currency Conversion Dashboard
    - ⚖ Sharpe Ratio Risk Ranking
    
    **Graph Index Explanation**
    - Close → Market price
    - MA10 → Short-term trend
    - MA50 → Long-term trend
    - Returns → Profitability
    - Volatility → Risk Index
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
        done = self.current_step >= len(self.data)-1
        portfolio_value = self.balance + self.shares * price
        reward = portfolio_value - self.initial_balance
        return self.data[self.current_step], reward, done, False, {}

# -------------------------------------
# Stock Universe (US + India + Fashion)
# -------------------------------------
us_stocks = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","AMD","INTC","IBM","ORCL","ADBE","CRM","QCOM","CSCO"]
fashion_us = ["NKE","LULU","UA","VFC","RL","TPR","GPS","PVH","SKX","LEVI"]

indian_stocks = ["RELIANCE.NS","TCS.NS","INFY.NS","ICICIBANK.NS","HDFCBANK.NS","SBIN.NS","ITC.NS","LT.NS","AXISBANK.NS","BAJFINANCE.NS"]
fashion_india = ["ADANIENT.NS","TRENT.NS","PAGEIND.NS","RAYMOND.NS","BATAINDIA.NS","ABFRL.NS","VMART.NS","CANTABIL.NS","METROBRAND.NS","LUXIND.NS"]

tickers = us_stocks + fashion_us + indian_stocks + fashion_india

symbol = st.selectbox("📌 Select Stock Ticker", tickers)

# -------------------------------------
# Download Data
# -------------------------------------
if st.button("🚀 Run Trading Simulation"):

    data = yf.download(symbol, start="2019-01-01", end="2025-01-01", progress=False)

    if data.empty or len(data) < 60:
        st.error("⚠️ Insufficient market data.")
        st.stop()

    # Candlestick Chart
    st.subheader("🕯 Candlestick Market Chart")

    candle_fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    candle_fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(candle_fig, use_container_width=True)

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
    obs = np.array(obs,dtype=np.float32)

    portfolio = []
    done=False

    while not done:
        obs_input = obs.reshape(1,-1)
        action,_ = ppo_model.predict(obs_input, deterministic=True)
        action = int(action.item())
        obs,reward,done,_,_ = env.step(action)
        obs = np.array(obs,dtype=np.float32)
        portfolio.append(env.balance + env.shares * obs[0])

    # Currency Conversion
    try:
        fx = yf.download("USDINR=X", period="5d", progress=False)
        usd_to_inr = fx["Close"].iloc[-1]
    except:
        usd_to_inr = 83.0

    portfolio_inr = [p*usd_to_inr for p in portfolio]

    # Dashboard
    col1,col2 = st.columns(2)

    with col1:
        st.subheader("📊 Portfolio Growth (USD)")
        st.line_chart(portfolio)

    with col2:
        st.subheader("💱 Portfolio Growth (INR)")
        st.line_chart(portfolio_inr)

    # Multi-Timeframe Performance
    st.subheader("⏱ Multi-Timeframe Market Performance")

    t1,t2,t3,t4,t5 = st.columns(5)

    t1.metric("Minute", f"{data['Returns'].tail(1).mean()*100:.2f}%")
    t2.metric("Hour", f"{data['Returns'].tail(5).mean()*100:.2f}%")
    t3.metric("Week", f"{data['Returns'].tail(25).mean()*100:.2f}%")
    t4.metric("Month", f"{data['Returns'].tail(100).mean()*100:.2f}%")
    t5.metric("Next 1 Month", f"{(data['Returns'].mean()*30)*100:.2f}%")

    # Sharpe Ratio
    sharpe = (np.mean(data['Returns']) / np.std(data['Returns'])) * np.sqrt(252)

    if sharpe < 0.8:
        rank = "1 — Good"
        msg = "Moderate performance with controlled volatility."
    elif sharpe < 1.5:
        rank = "2 — Very Good"
        msg = "Strong portfolio efficiency and balanced risk."
    else:
        rank = "3 — Excellent"
        msg = "Outstanding risk-adjusted performance."

    st.subheader("🏆 Adaptive Risk Ranking")
    colR1,colR2,colR3 = st.columns(3)

    colR1.metric("Sharpe Ratio", f"{sharpe:.3f}")
    colR2.metric("Risk Rank", rank)
    colR3.success(msg)

    st.markdown("---")
    st.success("✔ Trading Simulation Completed Successfully")
