# src/data_loader.py

import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import BDay
from functools import lru_cache

@lru_cache(maxsize=None)
def load_price_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker via yfinance.
    Returns a DataFrame with columns:
      - Open, High, Low, Close, Adj Close, Volume
      - Returns: simple pct change on Adj Close
    """
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, actions=False)

    # if columns are a MultiIndex (e.g. (ticker, field)), drop the top level:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ensure we have an Adj Close
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    df.dropna(how="all", inplace=True)
    df["Returns"] = df["Adj Close"].pct_change()
    df.dropna(subset=["Returns"], inplace=True)

    return df

@lru_cache(maxsize=None)
def load_vix(
    start: str,
    end: str,
    interval: str = "1d"
) -> pd.Series:
    """
    Download the VIX index (ticker '^VIX') and return its daily Adj Close.
    """
    vix_df = yf.download("^VIX", start=start, end=end, interval=interval,
        auto_adjust=False,
        actions=False)
    
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)

    if "Adj Close" in vix_df.columns:
        price_col = "Adj Close"
    elif "Close" in vix_df.columns:
        price_col = "Close"
    else:
        raise ValueError(f"VIX download missing Close/Adj Close columns: {vix_df.columns.tolist()}")

    vix_df.dropna(how="all", inplace=True)
    series = vix_df[price_col].pct_change() * 0 + vix_df[price_col]  # keep level values

    return series.rename("VIX")

def merge_price_and_vix(
    price_df: pd.DataFrame,
    vix_series: pd.Series
) -> pd.DataFrame:
    """
    Aligns price_df and vix_series on the same dates,
    forward‑fills any missing VIX values, and returns a
    single DataFrame.
    """
    df = price_df.join(vix_series, how="left")
    df["VIX"].ffill(inplace=True)
    return df

def load_implied_vol(ticker: str, as_of: str, horizon_days: int) -> float:
    """
    Returns the average ATM implied vol for options expiring
    closest to `as_of + horizon_days` via yfinance.
    """
    t = yf.Ticker(ticker)
    # target expiration date
    target = pd.to_datetime(as_of) + BDay(horizon_days)
    # find the soonest expiry ≥ target
    exps = [pd.to_datetime(d) for d in t.options]
    if not exps:
        raise ValueError(f"No option expirations found for {ticker}")
    
    exp = min([d for d in exps if d >= target], default=exps[-1])
    # pull option chain
    chain = t.option_chain(exp.strftime("%Y-%m-%d"))
    calls = chain.calls
    
    # find the strike closest to spot (ATM)
    spot = t.history(period='1d')['Close'].iloc[-1]
    calls['strike_diff'] = (calls['strike'] - spot).abs()
    atm_row = calls.loc[calls['strike_diff'].idxmin()]

    return float(atm_row['impliedVolatility'])