import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))
from src.utils.data_loader import fetch_data

def prepare_data(series, window_size=60):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

df = fetch_data("AAPL", interval="1m", period="7d")

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

X, y = prepare_data(scaled_close, window_size=60)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([
    LSTM(64, input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss=MeanSquaredError())
model.fit(X, y, epochs=10, batch_size=64, verbose=1)

os.makedirs("models", exist_ok=True)
model.save("models/price_model.h5")
joblib.dump(scaler, "models/scaler.pkl")
