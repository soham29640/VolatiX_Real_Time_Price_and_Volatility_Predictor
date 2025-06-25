import yfinance as yf
import pandas as pd

def fetch_data(ticker, interval="1m", period="1d"):
    df = yf.download(ticker, interval=interval, period=period, progress=False)

    if df.empty:
        raise ValueError(f"No data fetched for ticker: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')

    df = df.reset_index()
    df.rename(columns={'Datetime': 'Date'}, inplace=True)

    return df
