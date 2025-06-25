import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from src.utils.data_loader import fetch_data
from src.utils.load_and_predict_price_model import predict_next_prices
from src.utils.load_and_predict_volatility_model import predict_next_volatility

st.set_page_config(page_title="VolatiX Dashboard", layout="wide")
st.title("📊 VolatiX: Real-Time Price & Volatility Prediction")

st_autorefresh(interval=60000, key="auto_refresh")

ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
window_size = st.sidebar.slider("Window Size", min_value=30, max_value=120, value=60)
horizon = st.sidebar.slider("Prediction Horizon (minutes)", min_value=5, max_value=30, value=10)

df = fetch_data(ticker)

fig = go.Figure(data=[
    go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Candlestick'
    )
])
fig.update_layout(title=f"Live Candlestick Chart: {ticker}", xaxis_rangeslider_visible=False)

price_forecast = predict_next_prices(df, window_size=window_size, horizon=horizon)
volatility_forecast = predict_next_volatility(horizon=horizon)

next_vol = volatility_forecast.iloc[0]
threshold = np.percentile(volatility_forecast, 75)
risk_level = "🔴 High Risk" if next_vol > threshold else "🟢 Low Risk"

st.plotly_chart(fig, use_container_width=True)
st.subheader(f"📈 Next {horizon}-Minute Price Forecast")
st.line_chart(price_forecast)

st.subheader("📉 Volatility Forecast")
st.line_chart(volatility_forecast)

st.metric("Predicted Volatility (Next Minute)", f"{next_vol:.6f}")
st.metric("Risk Level", risk_level)

