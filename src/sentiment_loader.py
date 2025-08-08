# src/sentiment_loader.py

import os
import torch
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline

def fetch_polygon_news(
    ticker: str,
    from_date: str,
    to_date: str,
    api_key: str
) -> pd.DataFrame:
    """
    Pull Polygon’s news endpoint for a ticker
    between from_date/to_date.  Extract the
    API’s built-in sentiment per article.
    """
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker,
        "published_utc.gte": f"{from_date}T00:00:00Z",
        "published_utc.lte": f"{to_date}T23:59:59Z",
        "limit": 1000,
        "sort": "published_utc",
        "apiKey": api_key,
    }
    r = requests.get(url, params=params)
    data = r.json()
    if r.status_code != 200 or "results" not in data:
        raise RuntimeError(f"Polygon News error: {data.get('error') or r.text}")
    rows = []
    for art in data["results"]:
        dt = art["published_utc"][:10]
        text = art.get("title","") + " " + art.get("description","")
        # sentiment is in art["insights"] (may be empty) or art["sentiment"] at top level
        sent = None
        if "insights" in art and art["insights"]:
            sent = art["insights"][0].get("sentiment")
        elif "sentiment" in art:
            sent = art["sentiment"]
        # map to numeric
        if sent == "positive":
            score =  1.0
        elif sent == "negative":
            score = -1.0
        else:
            score = 0.0
        rows.append({"date": dt, "text": text, "sent_score": score})
    return pd.DataFrame(rows)

def daily_sentiment(df: pd.DataFrame) -> pd.Series:
    """
    Given df with ['date','text','sent_score'], group by date
    and return the mean sentiment score per day.
    """
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["sent_score"].mean().rename("NewsSent")
    return daily