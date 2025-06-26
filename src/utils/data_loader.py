import yfinance as yf
import pandas as pd

def fetch_data(ticker, interval="1m", period="1d"):
    df = yf.download(ticker, interval=interval, period=period, progress=False)

    if df.empty:
        raise ValueError(f"No data fetched for ticker: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

    df = df.reset_index()

    if 'Datetime' in df.columns:
        if df['Datetime'].dt.tz is None:
            df['Date'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        else:
            df['Date'] = df['Datetime'].dt.tz_convert('Asia/Kolkata')
        df.drop(columns=['Datetime'], inplace=True)
    else:
        df['Date'] = df.index.tz_convert('Asia/Kolkata')

    return df
