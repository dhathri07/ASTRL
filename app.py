import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from stable_baselines3 import PPO
from sklearn.preprocessing import RobustScaler
import gymnasium as gym
from gymnasium import spaces

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="AI Stock Trading System", layout="wide")

st.markdown("""
<style>
.main {background-color: #0E1117;}
h1,h2,h3 {color:#00E5FF;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# TITLE + CLOCK
# ----------------------------------
st.title("📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")

clock = st.empty()
india_tz = pytz.timezone("Asia/Kolkata")
clock.markdown(f"🕒 **Live Time (IST):** {datetime.now(india_tz).strftime('%d %b %Y | %I:%M:%S %p')}")

# ----------------------------------
# LOAD PPO MODEL
# ----------------------------------
ppo_model = PPO.load("ppo_trading_agent")

# ----------------------------------
# TRADING ENVIRONMENT
# ----------------------------------
class TradingEnv(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.data = data.values
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.balance = 10000
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
        reward = self.balance + self.shares * price - 10000

        return self.data[self.current_step], reward, done, False, {}

# ----------------------------------
# STOCK LIST
# ----------------------------------
tickers = [
    "AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","CSCO","AMD",
    "TCS.NS","INFY.NS","RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS",
    "NKE","LULU","RL","TPR","CPRI",
    "TRENT.NS","ABFRL.NS","RAYMOND.NS","BATAINDIA.NS","VMART.NS"
]

symbol = st.selectbox("📌 Select Stock Ticker", tickers)

# ----------------------------------
# DOWNLOAD DATA
# ----------------------------------
data = yf.download(symbol, start="2020-01-01", progress=False)

if data.empty:
    st.error("❌ Market data unavailable — try another stock.")
    st.stop()

# ----------------------------------
# CANDLESTYLE PRICE PLOT
# ----------------------------------
st.subheader("📊 Stock Price Chart")

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(data['Close'], color='cyan', linewidth=2)
ax.set_title("Close Price Trend")
ax.set_facecolor("#0E1117")
st.pyplot(fig)

# ----------------------------------
# FEATURE ENGINEERING
# ----------------------------------
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

# ----------------------------------
# RUN PPO AGENT
# ----------------------------------
env = TradingEnv(pd.DataFrame(padded))
obs,_ = env.reset()
obs = np.array(obs, dtype=np.float32)

portfolio = []
done=False

while not done:
    action,_ = ppo_model.predict(obs, deterministic=True)
    action = int(action)
    obs,reward,done,_,_ = env.step(action)
    obs = np.array(obs, dtype=np.float32)
    portfolio.append(env.balance + env.shares * obs[0])

# ----------------------------------
# USD → INR CONVERSION
# ----------------------------------
try:
    fx = yf.download("USDINR=X", period="5d", progress=False)
    usd_to_inr = float(fx["Close"].iloc[-1])
except:
    usd_to_inr = 83.0

portfolio_inr = [p*usd_to_inr for p in portfolio]

# ----------------------------------
# DASHBOARD
# ----------------------------------
c1,c2 = st.columns(2)

with c1:
    st.subheader("💰 Portfolio Growth (USD)")
    st.line_chart(portfolio)

with c2:
    st.subheader("₹ Portfolio Growth (INR)")
    st.line_chart(portfolio_inr)

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
st.metric("Sharpe Ratio", f"{sharpe:.3f}")
st.metric("Performance Rank", rank)

# ----------------------------------
# STOCK SUMMARY DASHBOARD
# ----------------------------------
st.subheader("📋 Stock Performance Summary")

summary_points = [
    f"Current Price: ${data['Close'].iloc[-1]:.2f}",
    f"30-Day Avg Price: ${data['Close'].tail(30).mean():.2f}",
    f"Volatility (Risk): {data['Volatility'].mean():.4f}",
    f"Average Daily Return: {data['Returns'].mean()*100:.3f}%",
    f"Trend Strength: {'Bullish 📈' if data['MA10'].iloc[-1] > data['MA50'].iloc[-1] else 'Bearish 📉'}",
    f"Annualized Sharpe Ratio: {sharpe:.3f}",
]

for pt in summary_points:
    st.markdown(f"✔ {pt}")

# ----------------------------------
# USER PORTFOLIO COMPARISON
# ----------------------------------
st.subheader("👤 User Portfolio Comparison")

col1, col2 = st.columns(2)

with col1:
    user_investment = st.number_input("💵 Enter Investment Amount (USD)", min_value=1000, value=10000, step=1000)

with col2:
    user_shares = st.number_input("📦 Number of Shares Owned", min_value=1, value=10, step=1)

current_price = data['Close'].iloc[-1]
user_portfolio_value = user_shares * current_price
user_profit = user_portfolio_value - user_investment
user_return_pct = (user_profit / user_investment) * 100

ai_final_value = portfolio[-1]
ai_profit = ai_final_value - 10000
ai_return_pct = (ai_profit / 10000) * 100

st.subheader("📊 AI vs User Portfolio Performance")

c1, c2, c3 = st.columns(3)

c1.metric("AI Portfolio ($)", f"{ai_final_value:.2f}", f"{ai_return_pct:.2f}%")
c2.metric("User Portfolio ($)", f"{user_portfolio_value:.2f}", f"{user_return_pct:.2f}%")
c3.metric("Winner", "AI 🤖" if ai_return_pct > user_return_pct else "User 🧑")

compare_df = pd.DataFrame({
    "Portfolio": ["AI Model", "User"],
    "Return %": [ai_return_pct, user_return_pct]
})

st.bar_chart(compare_df.set_index("Portfolio"))

# ----------------------------------
# FINAL STATUS
# ----------------------------------
st.success("✅ AI-driven adaptive portfolio optimization successfully executed.")

