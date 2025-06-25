import joblib
import numpy as np

def predict_next_volatility(horizon=10):
    model = joblib.load("models/volatility_model.pkl")
    forecast = model.forecast(horizon=horizon)
    forecasted_variance = forecast.variance.iloc[-1]
    forecasted_volatility = np.sqrt(forecasted_variance)
    return forecasted_volatility
