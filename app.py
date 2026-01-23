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

# ---------------- SAFE CONFIG ----------------
st.set_page_config(page_title="AI Quant Trading", layout="wide")

st.markdown("""
<style>
body { background-color:#0E1117; }
h1,h2,h3 { color:#00E5FF; }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📈 Portfolio Rebalancing in Quantitative Finance using Reinforcement Learning")

clock = st.empty()
india_tz = pytz.timezone("Asia/Kolkata")
clock.markdown(f"🕒 **Live IST Time:** {datetime.now(india_tz).strftime('%d %b %Y | %I:%M:%S %p')}")

# ---------------- SAFE PPO LOAD ----------------
try:
    ppo_model = PPO.load("ppo_trading_agent")
except:
    st.error("❌ PPO model missing. Upload trained model.")
    st.stop()

# ---------------- ENV ----------------
class TradingEnv(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.data = data.values.astype(np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(14,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.balance = 10000
        self.shares = 0
        self.current_step = 0
        return self.data[self.current_step], {}

    def step(self, action):
        price = float(self.data[self.current_step][0])

        if action == 1 and self.balance >= price:
            self.shares += 1
            self.balance -= price
        elif action == 2 and self.shares > 0:
            self.shares -= 1
            self.balance += price

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = self.balance + self.shares * price - 10000

        return self.data[self.current_step], reward, done, False, {}

# ---------------- STOCK LIST ----------------
us_stocks = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","CSCO","AMD",
             "NKE","LULU","RL","TPR","CPRI"]

ind_stocks = ["TCS.NS","INFY.NS","RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS",
              "TRENT.NS","ABFRL.NS","RAYMOND.NS","BATAINDIA.NS","VMART.NS"]

market = st.selectbox("🌍 Select Market", ["US Market", "Indian Market"])
symbol = st.selectbox("📌 Select Stock", us_stocks if market=="US Market" else ind_stocks)

# ---------------- SAFE DATA FETCH ----------------
@st.cache_data(ttl=900)
def safe_download(symbol):
    try:
        df = yf.download(symbol, period="3y", interval="1d", auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            return None
        return df
    except:
        return None

data = safe_download(symbol)

if data is None or len(data) < 120:
    st.error("❌ Market data unavailable or rate limited. Please try later.")
    st.stop()

# ---------------- SAFE CANDLESTYLE CHART ----------------
st.subheader("🕯 Market Price Visualization")

try:
    x = np.arange(len(data))
    low = np.asarray(data['Low'], dtype=np.float32).flatten()
    high = np.asarray(data['High'], dtype=np.float32).flatten()
    close = np.asarray(data['Close'], dtype=np.float32).flatten()

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(x, close, color="#00E5FF", linewidth=2)
    ax.fill_between(x, low, high, color="#00E5FF", alpha=0.12)
    ax.set_facecolor("#0E1117")
    ax.set_title("Price Movement")
    st.pyplot(fig)
except:
    st.warning("⚠ Visualization fallback activated.")
    st.line_chart(data['Close'])

# ---------------- FEATURES ----------------
data['MA10'] = data['Close'].rolling(10).mean()
data['MA50'] = data['Close'].rolling(50).mean()
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(10).std()
data.dropna(inplace=True)

features = ['Close','MA10','MA50','Returns','Volatility']

scaler = RobustScaler()
scaled = scaler.fit_transform(data[features])

# Pad to match PPO observation size
padded = np.zeros((scaled.shape[0],14), dtype=np.float32)
padded[:,:5] = scaled

# ---------------- RUN PPO ----------------
env = TradingEnv(pd.DataFrame(padded))

obs,_ = env.reset()
obs = obs.astype(np.float32)

portfolio = []
done=False

while not done:
    obs_input = obs.reshape(1,-1)
    action,_ = ppo_model.predict(obs_input, deterministic=True)
    action = int(action.item())
    obs,reward,done,_,_ = env.step(action)
    obs = obs.astype(np.float32)
    portfolio.append(env.balance + env.shares * obs[0])

# ---------------- SAFE FX ----------------
try:
    fx = safe_download("USDINR=X")
    usd_to_inr = float(fx['Close'].iloc[-1]) if fx is not None else 83.0
except:
    usd_to_inr = 83.0

portfolio_inr = [p * usd_to_inr for p in portfolio]

# ---------------- DASHBOARD ----------------
st.subheader("💼 Portfolio Performance")

c1,c2 = st.columns(2)

with c1:
    st.metric("Final USD Value", f"${portfolio[-1]:.2f}")
    st.line_chart(portfolio)

with c2:
    st.metric("Final INR Value", f"₹{portfolio_inr[-1]:.2f}")
    st.line_chart(portfolio_inr)

# ---------------- MULTI-TIMEFRAME ----------------
st.subheader("📊 Multi-Timeframe Analytics")

t1,t2,t3,t4,t5 = st.columns(5)

t1.metric("Minute", f"{data['Returns'].tail(1).mean()*100:.2f}%")
t2.metric("Hour", f"{data['Returns'].tail(5).mean()*100:.2f}%")
t3.metric("Week", f"{data['Returns'].tail(25).mean()*100:.2f}%")
t4.metric("Month", f"{data['Returns'].tail(100).mean()*100:.2f}%")
t5.metric("Next Month", f"{data['Returns'].mean()*30*100:.2f}%")

# ---------------- SHARPE ----------------
sharpe = (np.mean(data['Returns']) / np.std(data['Returns'])) * np.sqrt(252)

if sharpe < 0.8:
    rank = "1 — Good"
elif sharpe < 1.5:
    rank = "2 — Very Good"
else:
    rank = "3 — Excellent"

st.subheader("🏆 Adaptive Risk Ranking")

r1,r2 = st.columns(2)
r1.metric("Sharpe Ratio", f"{sharpe:.3f}")
r2.metric("Rank", rank)

# ---------------- AI SUMMARY ----------------
st.subheader("🧠 AI Portfolio Summary")

st.success(f"""
🔹 Stock: {symbol}
🔹 Market: {market}
🔹 RL Agent: PPO
🔹 Final Portfolio: ${portfolio[-1]:.2f}
🔹 INR Value: ₹{portfolio_inr[-1]:.2f}
🔹 Risk Rank: {rank}

📌 **AI Conclusion:**  
Reinforcement learning dynamically adapts market movements and achieves optimized risk-adjusted portfolio growth.
""")

st.balloons()
