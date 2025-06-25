import numpy as np
import joblib
from tensorflow.keras.models import load_model

def predict_next_prices(df, window_size=60, horizon=10):
    scaler = joblib.load("models/scaler.pkl")
    model = load_model("models/price_model.h5", compile=False)


    close_prices = df['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(close_prices)
    inputs = scaled[-window_size:].reshape(1, window_size, 1)

    predictions = []
    for _ in range(horizon):
        next_pred = model.predict(inputs, verbose=0)[0][0]
        predictions.append(next_pred)
        inputs = np.append(inputs[:, 1:, :], [[[next_pred]]], axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predicted_prices
