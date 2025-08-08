# src/features.py

import pandas as pd
import numpy as np
from pandas import Series

def make_multioutput_samples(
    feats: pd.DataFrame,
    horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given feats with column 'HistVol' as target,
    returns X (n_samples × n_features) and
    Y (n_samples × horizon) where
      Y[i, j] = HistVol at time (i + 1 + j)
    """
    Xs, Ys = [], []
    # i + 1 + horizon to exist for every i
    for i in range(len(feats) - horizon):
        Xs.append(feats.iloc[i].drop("HistVol").values)
        Ys.append(feats["HistVol"].iloc[i+1 : i+1+horizon].values)
    return np.array(Xs), np.array(Ys)

def compute_historical_volatility(
    returns: pd.Series,
    window: int = 21,
    annualization: int = 252
) -> pd.Series:
    """
    Rolling standard deviation of returns, annualized.
    Default window=21 trading days (~1 month).
    """
    return returns.rolling(window).std() * np.sqrt(annualization)

def create_lagged_returns(
    returns: pd.Series,
    lags: list[int] = [1, 5, 21]
) -> pd.DataFrame:
    """
    Generate lagged return features.
    e.g. Return_1 = yesterday’s return, Return_5 = 5-day avg return, etc.
    """
    df = pd.DataFrame(index=returns.index)
    for lag in lags:
        df[f"Ret_lag_{lag}"] = returns.shift(lag)
    return df

def create_rolling_features(
    df: pd.DataFrame,
    vol_windows: list[int] = [5, 21, 63],
    vol_annual: int = 252
) -> pd.DataFrame:
    """
    Add rolling volatility and volume‐based features:
      - Volatility_n: rolling std of returns over n days
      - Volume_avg_n : rolling mean of volume over n days
    """
    feats = pd.DataFrame(index=df.index)
    for w in vol_windows:
        feats[f"Vol_{w}d"]    = df["Returns"].rolling(w).std() * np.sqrt(vol_annual)
        feats[f"VolAvg_{w}d"] = df["Volume"].rolling(w).mean()
        feats[f"VolRel_{w}d"] = df["Volume"] / df["Volume"].rolling(w).mean()
    return feats

def assemble_features(
    price_df: pd.DataFrame,
    vix: pd.Series,
    ret_lags: list[int] = [1, 5, 21],
    vol_windows: list[int] = [5, 21, 63],
    news_sentiment: Series = None,
    technicals: dict[str, pd.Series] | None = None,
    macros: dict[str, pd.Series] | None = None
) -> pd.DataFrame:
    
    # Base features
    returns = price_df["Returns"]
    feats = pd.DataFrame(index=price_df.index)
    feats["HistVol"] = compute_historical_volatility(returns, window=vol_windows[1])
    feats = feats.join(create_lagged_returns(returns, lags=ret_lags))
    feats = feats.join(create_rolling_features(price_df, vol_windows=vol_windows))
    feats = feats.join(vix.rename("VIX"), how="left")

    if news_sentiment is not None:
        # align to feature dates, fill forward/back
        s = news_sentiment.reindex(feats.index).ffill().fillna(0.0)
        feats["NewsSent_lag_1"]  = s.shift(1)
        feats["NewsSent_lag_3"]  = s.shift(3)
        feats["NewsSent_roll_7"] = s.rolling(7).mean()
        feats["NewsVol_7"]       = s.rolling(7).count()

    if technicals is not None:
        for name, series in technicals.items():
            s = series.reindex(feats.index).ffill().bfill()
            feats[name]            = s
            feats[f"{name}_lag1"]  = s.shift(1)
            feats[f"{name}_roll5"] = s.rolling(5).mean()

    if macros is not None:
        for name, series in macros.items():
            s = series.reindex(feats.index).ffill().bfill()
            # raw level
            feats[name] = s
            # day-over-day pct change
            feats[f"{name}_chg"] = s.pct_change()
            # yesterday’s value
            feats[f"{name}_lag1"] = s.shift(1)

    # Final cleanup
    feats.dropna(inplace=True)
    return feats


