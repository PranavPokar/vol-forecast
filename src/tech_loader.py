# src/tech_loader.py
import os
import requests
import pandas as pd
from datetime import date

# POLY_KEY = os.getenv("POLYGON_API_KEY")
# BASE_URL = "https://api.polygon.io"

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the 14-period RSI (default) on a price series.
    Returns a Series indexed the same as `prices`.
    """
    delta = prices.diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)

    # use Wilder’s smoothing: EMA with com=window-1, adjust=False
    ma_up   = up.ewm(com=window-1, adjust=False).mean()
    ma_down = down.ewm(com=window-1, adjust=False).mean()

    rs  = ma_up / ma_down
    rsi = 100 - (100/(1 + rs))
    return rsi


def compute_macd(
    prices: pd.Series,
    short_window:  int = 12,
    long_window:   int = 26,
    signal_window: int = 9
) -> pd.DataFrame:
    """
    Compute MACD line and signal line on a price series.
    Returns a DataFrame with columns ['macd','signal','hist'].
    """
    ema_short = prices.ewm(span=short_window, adjust=False).mean()
    ema_long  = prices.ewm(span=long_window, adjust=False).mean()

    macd       = ema_short - ema_long
    macd_signal= macd.ewm(span=signal_window, adjust=False).mean()
    hist       = macd - macd_signal

    df = pd.DataFrame({
        "macd":   macd,
        "signal": macd_signal,
        "hist":   hist
    })
    return df