import yfinance as yf
import pandas as pd

def fetch_data(ticker, interval="1m", period="1d"):
    df = yf.download(ticker, interval=interval, period=period, progress=False)

    if df.empty:
        df = yf.download(ticker, interval="5m", period="5d", progress=False)
        if df.empty:
            raise ValueError(f"No data fetched for ticker: {ticker} (1m and 5m intervals failed)")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

    df = df.reset_index()

    if 'Datetime' in df.columns:
        dt_col = df['Datetime']
    else:
        dt_col = df.index.to_series()

    if dt_col.dt.tz is None:
        df['Date'] = dt_col.dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    else:
        df['Date'] = dt_col.dt.tz_convert('Asia/Kolkata')

    return df
