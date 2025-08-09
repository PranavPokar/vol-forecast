# src/fred_loader.py
import os
import pandas as pd
from pandas_datareader import data as pdr
from fredapi import Fred

# convenience wrapper to pull one series from FRED
def fetch_fred_series(series_id: str, start: str, end: str, api_key: str) -> pd.Series:
    """
    Fetch a FRED series by its ID, then reindex to business-day calendar.
    """
    df = pdr.DataReader(
        series_id,
        "fred",
        start=start,
        end=end,
        api_key=api_key
    )
    # DataReader returns a DataFrame with one column named series_id
    s = df[series_id].rename(series_id)
    # forward-fill from release date to next release
    bd = pd.date_range(start=s.index.min(), end=s.index.max(), freq="B")
    return s.reindex(bd).ffill().bfill()