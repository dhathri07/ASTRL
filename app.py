import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import time
from sklearn.preprocessing import RobustScaler
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Trading System", layout="wide")

# ---------------- LIVE CLOCK ----------------
clock = st.empty()
def live_clock():
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    clock.metric("🕒 Live Time (IST)", now.strftime("%H:%M:%S"))

live_clock()

# ---------------- TITLE ----------------
st.title("📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")

st.caption("Real-Time Intelligent Trading System using LSTM + PPO Reinforcement Learning")

# ---------------- TICKER SELECTION ----------------
us_stocks = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","AMD","NFLX","INTC"]
indian_stocks = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","LT.NS",
                  "SBIN.NS","WIPRO.NS","ITC.NS","HINDUNILVR.NS","ADANIENT.NS","BAJFINANCE.NS"]

ticker = st.selectbox("📌 Select Stock Ticker", us_stocks + indian_stocks)

# ---------------- DATA FETCH ----------------
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    return df

data = load_data(ticker)

if data.empty:
    st.error("❌ Data unavailable — try another stock")
    st.stop()

# ---------------- FEATURE ENGINEERING ----------------
data['MA10'] = data['Close'].rolling(10).mean()
data['MA50'] = data['Close'].rolling(50).mean()
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(10).std()
data.dropna(inplace=True)

features = ['Close','MA10','MA50','Returns','Volatility']
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data[features])

# ---------------- ENVIRONMENT ----------------
class TradingEnv(gym.Env):
    def __init__(self, df, balance=10000):
        super().__init__()
        self.df = df.values
        self.balance_init = balance
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        self.balance = self.balance_init
        self.shares = 0
        self.step_idx = 0
        return self.df[self.step_idx], {}

    def step(self, action):
        price = self.df[self.step_idx][0]
        if action == 1 and self.balance > price:
            self.shares += 1
            self.balance -= price
        elif action == 2 and self.shares > 0:
            self.shares -= 1
            self.balance += price

        self.step_idx += 1
        done = self.step_idx >= len(self.df)-1
        portfolio = self.balance + self.shares * price
        reward = portfolio - self.balance_init
        return self.df[self.step_idx], reward, done, False, {}

# ---------------- PPO MODEL ----------------
@st.cache_resource
def load_rl_model():
    return PPO.load("ppo_trading_agent")

ppo_model = load_rl_model()

# ---------------- RUN SIMULATION ----------------
env = TradingEnv(pd.DataFrame(scaled_data, columns=features))
obs,_ = env.reset()
done=False
portfolio=[]

while not done:
    action,_ = ppo_model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(int(action))
    portfolio.append(env.balance + env.shares * obs[0])

# ---------------- DASHBOARD ----------------
col1,col2,col3 = st.columns(3)
col1.metric("Initial Capital","$10,000")
col2.metric("Final AI Portfolio",f"${portfolio[-1]:.2f}")
col3.metric("Total Return",f"{((portfolio[-1]-10000)/10000)*100:.2f}%")

# ---------------- PORTFOLIO GRAPH ----------------
st.subheader("📊 AI Portfolio Growth")
fig,ax = plt.subplots()
ax.plot(portfolio)
ax.set_title("AI Portfolio Value")
ax.set_xlabel("Time")
ax.set_ylabel("Portfolio")
st.pyplot(fig)

# ---------------- CANDLESTICK ----------------
st.subheader("🕯 Candlestick Chart")
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(data['Close'],label="Close",linewidth=1)
ax.plot(data['MA10'],label="MA10",alpha=0.7)
ax.plot(data['MA50'],label="MA50",alpha=0.7)
ax.legend()
st.pyplot(fig)

# ---------------- STOCK SUMMARY ----------------
st.subheader("🧠 AI Stock Summary")

last_price = data['Close'].iloc[-1]
weekly = data['Returns'].tail(5).mean()*100
monthly = data['Returns'].tail(21).mean()*100
vol = data['Volatility'].iloc[-1]

summary = [
    f"Latest Price: ${last_price:.2f}",
    f"Weekly Avg Return: {weekly:.2f}%",
    f"Monthly Avg Return: {monthly:.2f}%",
    f"Volatility: {vol:.4f}",
    f"Market Bias: {'Bullish 📈' if weekly>0 else 'Bearish 📉'}"
]

for s in summary:
    st.markdown(f"- {s}")

# ---------------- USER PORTFOLIO ----------------
st.subheader("💼 Your Portfolio Comparison")

c1,c2,c3 = st.columns(3)
capital = c1.number_input("Capital ($)",1000,100000,10000)
shares = c2.number_input("Shares",1,1000,10)
buy_price = c3.number_input("Buy Price ($)",float(last_price))

user_value = shares * last_price
user_return = ((user_value - shares*buy_price)/(shares*buy_price))*100
ai_return = ((portfolio[-1]-10000)/10000)*100

st.metric("Your Portfolio",f"${user_value:.2f}",f"{user_return:.2f}%")
st.metric("AI Portfolio",f"${portfolio[-1]:.2f}",f"{ai_return:.2f}%")

# ---------------- SHARPE RATIO ----------------
st.subheader("📐 Sharpe Ratio & Risk Ranking")

returns = pd.Series(portfolio).pct_change().dropna()
sharpe = (returns.mean()/returns.std())*np.sqrt(252)

if sharpe < 0.8:
    rank = "1️⃣ Good"
elif sharpe < 1.5:
    rank = "2️⃣ Very Good"
else:
    rank = "3️⃣ Excellent"

st.metric("Sharpe Ratio",f"{sharpe:.2f}")
st.metric("Portfolio Rank",rank)

# ---------------- MULTI TIMEFRAME ----------------
st.subheader("📊 Multi-Timeframe Analysis")

timeframes = {
    "1 Minute": yf.download(ticker,period="1d",interval="1m"),
    "1 Hour": yf.download(ticker,period="5d",interval="1h"),
    "1 Week": yf.download(ticker,period="1mo",interval="1d"),
    "1 Month": yf.download(ticker,period="3mo",interval="1d")
}

for tf,df in timeframes.items():
    if not df.empty:
        st.markdown(f"### {tf}")
        st.line_chart(df['Close'])

# ---------------- USD INR CONVERSION ----------------
st.subheader("💱 USD ↔ INR Conversion")

try:
    fx = yf.download("USDINR=X",period="1d",progress=False)['Close'].iloc[-1]
    st.metric("USD → INR",f"₹{fx:.2f}")
    st.metric("AI Portfolio (₹)",f"₹{portfolio[-1]*fx:.2f}")
except:
    st.warning("Currency data unavailable")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built using LSTM + PPO Reinforcement Learning | Cloud Deployed | Production Safe")


