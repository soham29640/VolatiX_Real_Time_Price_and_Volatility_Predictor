import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from arch import arch_model
from src.utils.data_loader import fetch_data
from src.utils.load_and_predict_price_model import predict_next_prices

st.set_page_config(page_title="VolatiX Dashboard", layout="wide")
st.title("📊 VolatiX: Real-Time Price & Volatility Prediction")

st_autorefresh(interval=60000, key="auto_refresh")

ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
window_size = 60
horizon = st.sidebar.slider("Prediction Horizon (minutes)", min_value=5, max_value=30, value=10)

try:
    df = fetch_data(ticker)

    fig = go.Figure(data=[
        go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Candlestick'
        )
    ])
    fig.update_layout(title=f"Live Candlestick Chart: {ticker}", xaxis_rangeslider_visible=False)

    price_array = predict_next_prices(df, window_size=window_size, horizon=horizon)
    time_index = pd.date_range(
        start=df['Date'].iloc[-1] + pd.Timedelta(minutes=1),
        periods=horizon,
        freq='T'
    )
    price_forecast = pd.Series(price_array, index=time_index)

    last_price = df['Close'].iloc[-1]
    min_price = price_forecast.min()
    max_price = price_forecast.max()
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=price_forecast.index,
        y=price_forecast.values,
        mode='lines',
        name='Price Forecast',
        marker=dict(size=6),
        hoverinfo='x+y'
    ))
    price_fig.update_layout(
        title=f"📈 Next {horizon}-Minute Price Forecast",
        yaxis=dict(range=[min_price * 0.995, max_price * 1.005]),
        hovermode='x unified'
    )

    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    garch = arch_model(log_returns * 100, vol='GARCH', p=1, q=1)
    model_fit = garch.fit(disp='off')
    forecast_vol = model_fit.forecast(horizon=horizon)
    vol_forecast = np.sqrt(forecast_vol.variance.values[-1])
    volatility_forecast = pd.Series(vol_forecast, index=time_index, name="Volatility")

    next_vol = volatility_forecast.iloc[0]
    threshold = np.percentile(volatility_forecast, 75)
    risk_level = "🔴 High Risk" if next_vol > threshold else "🟢 Low Risk"

    vol_fig = go.Figure()
    vol_fig.add_trace(go.Scatter(
        x=volatility_forecast.index,
        y=volatility_forecast.values,
        mode='lines',
        name='Volatility Forecast',
        marker=dict(size=6),
        hoverinfo='x+y'
    ))
    vol_fig.update_layout(
        title="📉 Volatility Forecast",
        yaxis=dict(range=[volatility_forecast.min() * 0.95, volatility_forecast.max() * 1.05]),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(price_fig, use_container_width=True)
    st.plotly_chart(vol_fig, use_container_width=True)

    st.metric("Predicted Volatility (Next Minute)", f"{next_vol:.6f}")
    st.metric("Risk Level", risk_level)

except Exception as e:
    st.error(f"Error: {e}")
