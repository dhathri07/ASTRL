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
st.set_page_config(
    page_title="Portfolio Rebalancing using RL",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------
# Live Auto Refresh Clock
# -------------------------------------
clock_container = st.empty()

def live_clock():
    while True:
        india_tz = pytz.timezone("Asia/Kolkata")
        now = datetime.now(india_tz)
        clock_container.markdown(f"🕒 **Live Time (IST): {now.strftime('%d %b %Y | %I:%M:%S %p')}**")
        time.sleep(1)

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
# Main Title
# -------------------------------------
st.title("📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")
st.markdown("### AI-based Stock Trading & Portfolio Optimization Dashboard")

# Run clock
live_clock()

# -------------------------------------
# Ticker Dropdown (US + INDIA NSE)
# -------------------------------------
tickers_us = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","AMD","INTC","IBM","ORCL","SAP","ADBE","CRM","PYPL","BABA","UBER","CSCO","QCOM"]

tickers_india = [
    "RELIANCE.NS","TCS.NS","INFY.NS","ICICIBANK.NS","HDFCBANK.NS","SBIN.NS","LT.NS","HINDUNILVR.NS","ITC.NS",
    "BAJFINANCE.NS","BHARTIARTL.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS","WIPRO.NS","ULTRACEMCO.NS",
    "TITAN.NS","ADANIENT.NS","NTPC.NS","POWERGRID.NS"
]

symbol = st.selectbox("📌 Select Stock Ticker", tickers_us + tickers_india)

# -------------------------------------
# Run Simulation
# -------------------------------------
if st.button("🚀 Run Trading Simulation"):

    data = yf.download(symbol, period="1y", interval="1d", progress=False)

    if data.empty:
        st.error("⚠️ Unable to fetch data.")
        st.stop()

    # Feature Engineering
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
    obs = np.array(obs,dtype=np.float32)
    done = False
    portfolio = []

    while not done:
        obs_input = obs.reshape(1,-1)
        action,_ = ppo_model.predict(obs_input, deterministic=True)
        action = int(action.item())
        obs,reward,done,_,_ = env.step(action)
        obs = np.array(obs,dtype=np.float32)
        portfolio.append(env.balance + env.shares * obs[0])

    # -------------------------------------
    # USD → INR Conversion
    # -------------------------------------
    usd_inr = yf.download("USDINR=X", period="1d", progress=False)["Close"][-1]
    final_usd = portfolio[-1]
    final_inr = final_usd * usd_inr

    # -------------------------------------
    # Dashboard Layout
    # -------------------------------------
    col1,col2,col3 = st.columns(3)

    col1.metric("💰 Final Portfolio (USD)", f"${final_usd:.2f}")
    col2.metric("₹ Final Portfolio (INR)", f"₹{final_inr:.2f}")
    col3.metric("📊 Return %", f"{((final_usd-10000)/10000)*100:.2f}%")

    # -------------------------------------
    # Multi-Timeframe Visualizations
    # -------------------------------------
    st.subheader("📈 Multi-Timeframe Stock Performance")

    tf1,tf2,tf3,tf4,tf5 = st.tabs(["Minute","Hourly","Daily","Weekly","Monthly"])

    def plot_tf(interval,label):
        df = yf.download(symbol, period="5d", interval=interval, progress=False)
        fig,ax = plt.subplots(figsize=(8,4))
        ax.plot(df['Close'])
        ax.set_title(label)
        ax.grid(True)
        st.pyplot(fig)

    with tf1: plot_tf("1m","Minute Trend")
    with tf2: plot_tf("1h","Hourly Trend")
    with tf3: plot_tf("1d","Daily Trend")
    with tf4: plot_tf("1wk","Weekly Trend")
    with tf5: plot_tf("1mo","Monthly Trend")

    # -------------------------------------
    # 1 Month Future Forecast (Statistical Projection)
    # -------------------------------------
    st.subheader("🔮 1 Month Future Price Projection")

    last_price = data['Close'].iloc[-1]
    returns_mean = data['Returns'].mean()
    future_prices = [last_price]

    for _ in range(30):
        future_prices.append(future_prices[-1] * (1 + returns_mean))

    fig,ax = plt.subplots(figsize=(10,4))
    ax.plot(future_prices, linestyle='--')
    ax.set_title("Projected 1 Month Price Trend")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.grid(True)
    st.pyplot(fig)

    st.success("✅ Trading Simulation & Forecast Completed Successfully!")
