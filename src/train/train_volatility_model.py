import os
import sys
import numpy as np
from arch import arch_model
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.data_loader import fetch_data

def prepare_data(series, window_size):
    series = np.asarray(series)
    recent = series[-(window_size + 1):]
    recent = np.clip(recent, 1e-8, None)
    log_data = np.log(recent[1:] / recent[:-1])
    return log_data

df = fetch_data("AAPL", interval="1m", period="5d")

y = prepare_data(df['Close'], window_size=60)

model = arch_model(y, vol='GARCH', p=1, q=1, mean='zero')
model_fit = model.fit(disp='off')

print(model_fit.summary())

os.makedirs("models", exist_ok=True)
joblib.dump(model_fit, "models/volatility_model.pkl")
