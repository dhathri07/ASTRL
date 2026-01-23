import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sklearn.preprocessing import RobustScaler
import gymnasium as gym
from gymnasium import spaces
import time
from datetime import datetime

# -------------------------------------
# Streamlit Page Config
# -------------------------------------
st.set_page_config(page_title="AI Trading RL System", layout="wide")

# -------------------------------------
# Main Title
# -------------------------------------
st.title("📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")

# -------------------------------------
# Live Clock
# -------------------------------------
clock_placeholder = st.empty()

def live_clock():
    clock_placeholder.markdown(f"### 🕒 Current Time: {datetime.now().strftime('%d %b %Y | %H:%M:%S')}")

live_clock()

# -------------------------------------
# Load PPO Model Safely
# -------------------------------------
@st.cache_resource
def load_rl_model():
    try:
        return PPO.load("ppo_trading_agent")
    except:
        return None

ppo_model = load_rl_model()

# -------------------------------------
# Stock Lists
# -------------------------------------
us_stocks = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","NFLX","INTC","AMD",
    "CSCO","ORCL","IBM","UBER","ADBE","PYPL","QCOM","AVGO","CRM","PEP"
]

indian_stocks = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","ITC.NS","BHARTIARTL.NS","LT.NS","KOTAKBANK.NS",
    "AXISBANK.NS","HINDUNILVR.NS","ASIANPAINT.NS","MARUTI.NS",
    "BAJFINANCE.NS","WIPRO.NS","ULTRACEMCO.NS","TITAN.NS","POWERGRID.NS","NTPC.NS"
]

# -------------------------------------
# Sidebar Controls
# -------------------------------------
st.sidebar.header("⚙️ Trading Controls")

market = st.sidebar.selectbox("Select Market", ["US Market", "Indian Market"])

if market == "US Market":
    ticker = st.sidebar.selectbox("Select Stock", us_stocks)
    currency = "USD"
else:
    ticker = st.sidebar.selectbox("Select Stock", indian_stocks)
    currency = "INR"

run_sim = st.sidebar.button("🚀 Run Trading Simulation")

# -------------------------------------
# Data Loader (Fail-safe)
# -------------------------------------
@st.cache_data(show_spinner=False)
def load_stock_data(symbol):
    try:
        df = yf.download(symbol, period="3y", interval="1d", progress=False)
        if df.empty:
            return None
        return df
    except:
        return None

# -------------------------------------
# Trading Environment
# -------------------------------------
class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100000):
        super().__init__()
        self.data = data.values.astype(np.float32)
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32
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
# Main Execution
# -------------------------------------
if run_sim:

    data = load_stock_data(ticker)

    if data is None:
        st.error("❌ Failed to fetch stock data. Yahoo rate-limit or network issue.")
        st.stop()

    st.subheader(f"📊 Market Data Overview — {ticker}")
    st.line_chart(data["Close"])

    # Feature Engineering
    data["MA10"] = data["Close"].rolling(10).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["Returns"] = data["Close"].pct_change()
    data["Volatility"] = data["Returns"].rolling(10).std()
    data.dropna(inplace=True)

    features = ["Close","MA10","MA50","Returns","Volatility"]

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data[features])

    env = TradingEnv(pd.DataFrame(scaled_data, columns=features))

    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32)

    done = False
    portfolio = []

    while not done:
        obs_input = obs.reshape(1, -1)

        if ppo_model is None:
            action = np.random.randint(0,3)
        else:
            action, _ = ppo_model.predict(obs_input, deterministic=True)
            action = int(action)

        obs, reward, done, _, _ = env.step(action)
        obs = np.array(obs, dtype=np.float32)

        portfolio.append(env.balance + env.shares * obs[0])

    st.subheader("📈 RL Trading Portfolio Growth")
    st.line_chart(portfolio)

    final_value = float(portfolio[-1])

    st.success(f"Final Portfolio Value: {currency} {final_value:,.2f}")

    # -------------------------------------
    # Multi-Timeframe Visualization
    # -------------------------------------
    st.subheader("📊 Multi-Timeframe Market Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Hourly Trend**")
        hourly = data["Close"].resample("H").mean()
        st.line_chart(hourly.tail(100))

    with col2:
        st.markdown("**Weekly Trend**")
        weekly = data["Close"].resample("W").mean()
        st.line_chart(weekly)

    with col3:
        st.markdown("**Monthly Trend**")
        monthly = data["Close"].resample("ME").mean()
        st.line_chart(monthly)

    # -------------------------------------
    # 30 Day Projection
    # -------------------------------------
    st.subheader("📆 Next 30-Day Trend Projection")

    forecast = data["Close"].tail(30).mean()
    future_projection = [forecast * (1 + np.random.normal(0.001,0.01)) for _ in range(30)]

    st.line_chart(future_projection)

    # -------------------------------------
    # INR Currency Dashboard (FAIL-SAFE)
    # -------------------------------------
    if currency == "INR":

        st.subheader("🇮🇳 Indian Market Currency Dashboard")

        try:
            usd_inr = yf.download("USDINR=X", period="5d", interval="1d", progress=False)

            if not usd_inr.empty and "Close" in usd_inr.columns:

                st.line_chart(usd_inr["Close"])

                latest_rate = float(usd_inr["Close"].iloc[-1])
                usd_equivalent = final_value / latest_rate

                colM, colN = st.columns(2)
                colM.metric("Portfolio (INR)", f"₹ {final_value:,.2f}")
                colN.metric("USD Equivalent", f"$ {usd_equivalent:,.2f}")

            else:
                st.warning("⚠️ USD-INR data unavailable. Currency conversion skipped.")

        except:
            st.warning("⚠️ Currency data temporarily unavailable due to rate limiting.")

# -------------------------------------
# Footer
# -------------------------------------
st.markdown("---")
st.markdown("### 🧠 AI Trading System • PPO Reinforcement Learning • Streamlit Cloud Deployment")
