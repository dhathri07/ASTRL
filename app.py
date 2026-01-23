import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# Load PPO Model ONLY (LSTM removed for stability)
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
# UI Header
# -------------------------------------
st.title("📈 AI Stock Trading System")
st.markdown("### Deep Reinforcement Learning based Automated Trading Dashboard")

# -------------------------------------
# Stock Selector (20+ stocks)
# -------------------------------------
tickers = [
    "AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","AMD","INTC",
    "IBM","ORCL","SAP","ADBE","CRM","PYPL","BABA","UBER","CSCO","QCOM"
]

symbol = st.selectbox("📌 Select Stock Ticker", tickers)

# -------------------------------------
# Run Simulation
# -------------------------------------
if st.button("🚀 Run Trading Simulation"):

    try:
        data = yf.download(symbol, start="2019-01-01", end="2025-01-01", progress=False)
    except:
        st.error("⚠️ Data source rate-limited. Try again later.")
        st.stop()

    if data.empty or len(data) < 60:
        st.error("⚠️ Insufficient market data.")
        st.stop()

    # Feature engineering
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(10).std()
    data.dropna(inplace=True)

    features = ['Close','MA10','MA50','Returns','Volatility']

    # Scaling
    scaler = RobustScaler()
    scaled = scaler.fit_transform(data[features])

    # FINAL FIX → Pad to 14 features for PPO compatibility
    padded = np.zeros((scaled.shape[0], 14))
    padded[:, :5] = scaled

    env = TradingEnv(pd.DataFrame(padded))

    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32)

    done = False
    portfolio = []

    # -------------------------------------
    # Trading Simulation
    # -------------------------------------
    while not done:
        obs_input = obs.reshape(1, -1)
        action, _ = ppo_model.predict(obs_input, deterministic=True)

        # FINAL FIX — safe scalar conversion
        action = int(action.item())

        obs, reward, done, _, _ = env.step(action)
        obs = np.array(obs, dtype=np.float32)

        portfolio.append(env.balance + env.shares * obs[0])

    # -------------------------------------
    # Dashboard Visualization
    # -------------------------------------
    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("📊 Portfolio Growth")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(portfolio, linewidth=2)
        ax.set_xlabel("Trading Steps")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True)
        st.pyplot(fig)

    with col2:
        st.subheader("📌 Performance Summary")
        st.metric("Initial Capital", "$10,000")
        st.metric("Final Portfolio", f"${portfolio[-1]:.2f}")
        st.metric("Net Profit", f"${portfolio[-1]-10000:.2f}")
        st.metric("Return %", f"{((portfolio[-1]-10000)/10000)*100:.2f}%")

    st.success("✅ Trading simulation completed successfully!")
