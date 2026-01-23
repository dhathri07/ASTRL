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
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------
# Title
# -------------------------------------
st.title("📊 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")
st.markdown("### AI-based Automated Trading & Real-Time Market Visualization")

# -------------------------------------
# Live Clock
# -------------------------------------
clock_placeholder = st.empty()

def live_clock():
    while True:
        india_tz = pytz.timezone("Asia/Kolkata")
        current_time = datetime.now(india_tz)
        clock_placeholder.markdown(f"🕒 **Current Time:** {current_time.strftime('%d %b %Y | %I:%M:%S %p')}")
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
# Stock Selection
# -------------------------------------
global_stocks = [
    "AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","AMD","INTC",
    "IBM","ORCL","SAP","ADBE","CRM","PYPL","BABA","UBER","CSCO","QCOM"
]

indian_stocks = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","ITC.NS",
    "LT.NS","HINDUNILVR.NS","BHARTIARTL.NS","KOTAKBANK.NS","AXISBANK.NS",
    "BAJFINANCE.NS","MARUTI.NS","SUNPHARMA.NS","WIPRO.NS","ASIANPAINT.NS",
    "TATAMOTORS.NS","ULTRACEMCO.NS","HCLTECH.NS"
]

st.sidebar.header("Market Selection")
market_type = st.sidebar.radio("Select Market", ["US Stocks", "Indian Stocks"])

if market_type == "US Stocks":
    symbol = st.sidebar.selectbox("Select Stock", global_stocks)
    currency = "USD"
else:
    symbol = st.sidebar.selectbox("Select Stock", indian_stocks)
    currency = "INR"

# -------------------------------------
# Run Simulation
# -------------------------------------
if st.sidebar.button("🚀 Run Trading Simulation"):

    data = yf.download(symbol, start="2021-01-01", end="2025-01-01", progress=False)

    if data.empty or len(data) < 60:
        st.error("⚠️ Insufficient data for this stock.")
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

    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32)

    done = False
    portfolio = []

    while not done:
        obs_input = obs.reshape(1, -1)
        action, _ = ppo_model.predict(obs_input, deterministic=True)
        action = int(action.item())
        obs, reward, done, _, _ = env.step(action)
        obs = np.array(obs, dtype=np.float32)
        portfolio.append(env.balance + env.shares * obs[0])

    # -------------------------------------
    # Dashboards
    # -------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Initial Capital", f"{currency} 10,000")
    col2.metric("Final Portfolio", f"{currency} {portfolio[-1]:.2f}")
    col3.metric("Return %", f"{((portfolio[-1]-10000)/10000)*100:.2f}%")

    st.subheader("📊 Portfolio Growth Curve")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(portfolio, linewidth=2)
    ax.set_ylabel("Portfolio Value")
    ax.set_xlabel("Trading Steps")
    ax.grid(True)
    st.pyplot(fig)

    # -------------------------------------
    # Multi-Timeframe Visualizations
    # -------------------------------------
    st.subheader("📈 Market Performance Visualization")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Minute Trend")
        minute = yf.download(symbol, period="1d", interval="1m")
        st.line_chart(minute["Close"])

        st.markdown("### Hourly Trend")
        hour = yf.download(symbol, period="5d", interval="60m")
        st.line_chart(hour["Close"])

    with colB:
        st.markdown("### Daily Trend")
        st.line_chart(data["Close"])

        st.markdown("### Weekly Trend")
        weekly = data["Close"].resample("W").mean()
        st.line_chart(weekly)

    # -------------------------------------
    # Monthly & Future Forecast
    # -------------------------------------
    st.subheader("📅 Monthly & Future Performance")

    colX, colY = st.columns(2)

    with colX:
        st.markdown("### Monthly Trend")
        monthly = data["Close"].resample("M").mean()
        st.line_chart(monthly)

    with colY:
        st.markdown("### Next 1-Month Forecast (Trend Extrapolation)")
        x = np.arange(len(data["Close"]))
        coef = np.polyfit(x, data["Close"], 1)
        future_x = np.arange(len(x), len(x)+30)
        future_price = coef[0]*future_x + coef[1]

        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.plot(data["Close"], label="Historical")
        ax2.plot(future_x, future_price, '--', label="Forecast")
        ax2.legend()
        st.pyplot(fig2)

    # -------------------------------------
    # INR Currency Dashboard
    # -------------------------------------
    if currency == "INR":
        st.subheader("🇮🇳 Indian Market Currency Dashboard")

        usd_inr = yf.download("USDINR=X", period="1mo", interval="1d")
        st.line_chart(usd_inr["Close"])

        inr_value = portfolio[-1]
        usd_equivalent = inr_value / usd_inr["Close"][-1]

        colM, colN = st.columns(2)
        colM.metric("Portfolio (INR)", f"₹ {inr_value:.2f}")
        colN.metric("USD Equivalent", f"$ {usd_equivalent:.2f}")

    st.success("✅ Trading Simulation & Visualization Completed Successfully!")

# -------------------------------------
# Start Live Clock
# -------------------------------------
live_clock()
