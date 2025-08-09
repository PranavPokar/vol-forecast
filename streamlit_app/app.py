# streamlit_app/app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from src.data_loader import load_implied_vol
from src.data_loader import load_price_data, load_vix, merge_price_and_vix
from src.features import make_multioutput_samples
from src.features import assemble_features
from src.models import (fit_garch,
                        train_rf_multioutput, 
                        train_nn_multioutput,
                        train_xgb_multioutput)

from src.sentiment_loader import (fetch_polygon_news, daily_sentiment)
from src.tech_loader import compute_macd, compute_rsi
from src.fred_loader import fetch_fred_series

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Volatility Forecasting", layout="wide")
st.title("📈 Volatility Forecasting Dashboard")

#Sidebar for user inputs
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="AAPL")
    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End date", value=date.today())
    vol_window = st.slider("Historical vol window (days)", min_value=5, max_value=63, value=21)
    model_choice = st.selectbox(
    "Model",
    ["GARCH", "RF-MO", "XGB-MO", "NN-MO"]
    )
    horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=5)
    history_days = st.slider("History window for chart (days)", min_value=30, max_value=365, value=180)

    # Hyperparameters
    if model_choice in ["RF-MO", "XGB-MO"]:
        n_estimators = st.number_input("n_estimators", min_value=10, max_value=1000, value=100)
        max_depth = st.number_input("max_depth", min_value=1, max_value=20, value=5)
        learning_rate  = st.number_input("learning_rate", 0.01, 1.0, 0.1, step=0.01)
        subsample      = st.number_input("subsample",     0.1, 1.0, 1.0, step=0.1)
        colsample_bytree = st.number_input("colsample_bytree", 0.1, 1.0, 1.0, step=0.1)
        reg_alpha      = st.number_input("reg_alpha",     0.0, 1.0, 0.0, step=0.01)
        reg_lambda     = st.number_input("reg_lambda",    0.0, 5.0, 1.0, step=0.1)
    if model_choice in ["NN-MO"]:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
        batch_size = st.number_input("Batch size", min_value=1, max_value=1024, value=32)

# Run Forecast
if st.sidebar.button("Run Forecast"):

    poly_key = st.secrets["POLYGON_API_KEY"]
    fred_key = st.secrets["FRED_API_KEY"]

    # sanity-check existence
    if not poly_key:
        st.error("Missing POLYGON_API_KEY in secrets")
        st.stop()
    if not fred_key:
        st.error("Missing FRED_API_KEY in secrets")
        st.stop()

    # Load data
    price_df = load_price_data(ticker, start_date.isoformat(), end_date.isoformat())
    vix = load_vix(start_date.isoformat(), end_date.isoformat())
    df = merge_price_and_vix(price_df, vix)

    cpi_ser = fetch_fred_series("CPIAUCSL", start_date.isoformat(), end_date.isoformat(), fred_key)
    unemp_ser= fetch_fred_series("UNRATE", start_date.isoformat(), end_date.isoformat(), fred_key)
    fedf_ser = fetch_fred_series("FEDFUNDS", start_date.isoformat(), end_date.isoformat(), fred_key)

    price_series = price_df["Adj Close"]
    rsi_series   = compute_rsi(price_series, window=14)
    macd_df      = compute_macd(price_series, short_window=12, long_window=26, signal_window=9)

    technicals = {
    "RSI": rsi_series,
    "MACD": macd_df["macd"],
    "MACD_signal": macd_df["signal"]
    }

    raw_news = fetch_polygon_news(
        ticker,
        start_date.isoformat(),
        end_date.isoformat(),
        poly_key
    )
    # daily_sentiment(...) returns a pd.Series named "NewsSent"
    news_sent = daily_sentiment(raw_news)
    if news_sent.empty:
        st.warning("No Polygon news found for that period; proceeding without sentiment")
        news_sent = None

    # Feature engineering WITHOUT sentiment
    feats = assemble_features(df, vix, news_sentiment=news_sent, technicals=technicals, macros={
            "CPI": cpi_ser,
            "Unemp": unemp_ser,
            "FedFunds": fedf_ser,
        })
    X     = feats.drop(columns=["HistVol"])
    y     = feats["HistVol"]

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Dates for plotting
    last_date    = feats.index[-1]
    future_dates = [last_date + BDay(i) for i in range(1, horizon+1)]
    hist_vol     = feats["HistVol"]

    # Forecast
    if model_choice == "GARCH":
        res    = fit_garch(df["Returns"].dropna().values)
        fcasts = res.forecast(horizon=horizon, reindex=False).variance.iloc[-1]
        fvol   = np.sqrt(fcasts.values)
        fvol = fvol * (hist_vol.iloc[-1] / fvol[0])

    elif model_choice == "RF-MO":
        X_mo, Y_mo = make_multioutput_samples(feats, horizon)
        if len(X_mo) < 1:
            st.error("Not enough data for RF-MO – shorten horizon or extend history.")
            st.stop()
        mo_rf = train_rf_multioutput(
            X_mo, Y_mo,
            n_estimators=n_estimators, max_depth=max_depth
        )
        last_X = X_mo[-1].reshape(1, -1)
        fvol   = mo_rf.predict(last_X)[0]

    elif model_choice == "XGB-MO":
        # build multi-output arrays
        X_mo, Y_mo = make_multioutput_samples(feats, horizon)
        if len(X_mo) < 1:
            st.error("Not enough data for XGB‑MO – shorten horizon or extend history.")
            st.stop()

        # train & predict using XGB-MO
        mo_xgb = train_xgb_multioutput(
            X_mo, Y_mo,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda
        )
        # most recent features vector
        last_X = X_mo[-1].reshape(1, -1)
        # fvol is a horizon‑length array
        fvol   = mo_xgb.predict(last_X)[0]

    else:  # NN‑MO
        X_mo, Y_mo = make_multioutput_samples(feats, horizon)
        if len(X_mo) < 1:
            st.error("Not enough data for NN‑MO – shorten horizon or extend history.")
            st.stop()
        # 1) scale X
        scaler = StandardScaler().fit(X_mo)
        X_mo_s = scaler.transform(X_mo)
        last_X  = X_mo[-1].reshape(1, -1)
        last_X_s = scaler.transform(last_X)

        # 2) train & predict
        mo_nn = train_nn_multioutput(
            X_mo_s, Y_mo,
            hidden_units=64,
            epochs=epochs,
            batch_size=batch_size
        )
        fvol = mo_nn.predict(last_X_s)[0]

    # Load IV and Signal
    forecast_series = pd.Series(fvol, index=future_dates, name="Forecast Vol")
    iv_today = load_implied_vol(ticker, last_date.isoformat(), horizon)
    signal = forecast_series - iv_today

    fig = go.Figure()
    # Historical Vol
    fig.add_trace(go.Scatter(
        x=hist_vol.index, y=hist_vol,
        name="Historical Vol", line=dict(color="blue", width=2)
    ))
    # Forecast Vol
    fig.add_trace(go.Scatter(
        x=forecast_series.index, y=forecast_series,
        name=f"Forecast {horizon}-day Vol",
        line=dict(color="red", dash="dash")
    ))
    # “Today” marker
    fig.add_vline(x=last_date, line=dict(color="gray", dash="dot"))
    fig.add_annotation(x=last_date, y=1, yref="paper",
                       text="Today", showarrow=False,
                       yanchor="bottom", xanchor="left")
    
    # Zoom to history_days + forecast
    start_plot = last_date - pd.Timedelta(days=history_days)
    fig.update_xaxes(range=[start_plot, forecast_series.index[-1]])
    fig.update_layout(
        title=f"{ticker}: Vol History & {horizon}-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Volatility (σ)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast-only
    st.subheader(f"Forecast-only ({horizon}-day Vol)")
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(
        x=forecast_series.index, y=forecast_series,
        name="Forecast", line=dict(color="red", width=3)
    ))
    fig_f.update_layout(
        title=f"{ticker} {horizon}-Day Vol Forecast",
        xaxis_title="Date", yaxis_title="Volatility"
    )
    st.plotly_chart(fig_f, use_container_width=True)

    # Expected move
    current_price = price_df["Adj Close"].iloc[-1]
    exp_pct  = forecast_series.iloc[-1] * np.sqrt(horizon / 252)
    exp_doll = exp_pct * current_price
    st.markdown(
        f"**Expected move over next {horizon} days:** "
        f"±{exp_pct:.2%} (±${exp_doll:,.2f} at ${current_price:.2f})"
    )

    st.subheader("News Sentiment (7d MA) vs. Historical Volatility")
    if news_sent is not None:
        df_drive = pd.DataFrame({
            "Volatility": hist_vol,
            "NewsSent":  news_sent.reindex(hist_vol.index).ffill()
        }).dropna()

        # smooth sentiment with a 7-day MA
        df_drive["Sent_7d"] = df_drive["NewsSent"].rolling(7, min_periods=1).mean()

        today_dt = last_date.to_pydatetime()

        # 4) build figure
        fig_drive = make_subplots(specs=[[{"secondary_y": True}]])

        # — historical vol (left axis)
        fig_drive.add_trace(
            go.Scatter(
                x=df_drive.index,
                y=df_drive["Volatility"],
                name="HistVol",
                line=dict(color="royalblue", width=2)
            ),
            secondary_y=False
        )

        # — smoothed sentiment (right axis)
        fig_drive.add_trace(
            go.Scatter(
                x=df_drive.index,
                y=df_drive["Sent_7d"],
                name="NewsSent (7d MA)",
                line=dict(color="orange", width=2)
            ),
            secondary_y=True
        )

        # — threshold lines as actual traces
        date_range = [df_drive.index[0], df_drive.index[-1]]
        fig_drive.add_trace(
            go.Scatter(
                x=date_range,
                y=[0.5, 0.5],
                mode="lines",
                name="Positive Sent Thr (+0.5)",
                line=dict(color="green", dash="dot")
            ),
            secondary_y=True
        )
        fig_drive.add_trace(
            go.Scatter(
                x=date_range,
                y=[-0.5, -0.5],
                mode="lines",
                name="Negative Sent Thr (–0.5)",
                line=dict(color="red", dash="dot")
            ),
            secondary_y=True
        )

        # — shade sustained positive-sentiment regimes
        mask = df_drive["Sent_7d"] > 0.5
        starts = df_drive.index[mask & ~mask.shift(1, fill_value=False)]
        ends   = df_drive.index[mask & ~mask.shift(-1, fill_value=False)]
        for x0, x1 in zip(starts, ends):
            fig_drive.add_vrect(
                x0=x0.to_pydatetime(),
                x1=x1.to_pydatetime(),
                fillcolor="green",
                opacity=0.1,
                line_width=0,
                layer="below"
            )

        # — Today line (as a shape) + annotation
        fig_drive.add_shape(
            type="line",
            x0=today_dt, x1=today_dt,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="gray", dash="dash"),
        )
        fig_drive.add_annotation(
            x=today_dt, y=1.0,
            xref="x", yref="paper",
            text="Today",
            showarrow=False,
            yanchor="bottom", xanchor="left",
            font=dict(color="gray")
        )

        # 5) layout tweaks
        fig_drive.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.15,
                xanchor="center", x=0.5
            ),
            margin=dict(t=80)
        )

        # — Vol axis never below zero
        fig_drive.update_yaxes(
            title_text="Volatility (σ)",
            secondary_y=False,
            range=[0, df_drive["Volatility"].max() * 1.1]
        )
        # — Sentiment axis fixed ±1
        fig_drive.update_yaxes(
            title_text="Sentiment Score",
            secondary_y=True,
            range=[-1, 1]
        )

        st.plotly_chart(fig_drive, use_container_width=True)

        # 6) concise explanation
        st.markdown(
            "**How to read this:**  \n"
            "- **HistVol** (blue) shows realized volatility σ over time.  \n"
            "- **NewsSent (7d MA)** (orange) is your 7-day average signed FinBERT sentiment.  \n"
            "- When sentiment > +0.5 (green dotted line → shaded bands), markets are very positive—vol often remains low but complacency risk builds.  \n"
            "- When sentiment < –0.5 (red dotted line), negativity dominates and volatility often rises next.  \n"
            "- The vertical “Today” line marks your forecast pivot point."
        )

        # build the conditional distribution behind the scenes
        sent7 = news_sent.reindex(hist_vol.index).ffill().rolling(7, min_periods=1).mean()
        cond = pd.concat([hist_vol, sent7], axis=1).dropna()
        cond.columns = ["HistVol","NewsSent7"]
        cond["FutVol"] = cond["HistVol"].shift(-horizon)
        cond = cond.dropna(subset=["FutVol"])
        # bucket by today’s sentiment bin
        bins = np.arange(-1.0, 1.05, 0.1)
        today_val = sent7.iloc[-1]
        today_bin = pd.cut([today_val], bins=bins)[0]
        bucket = cond[pd.cut(cond["NewsSent7"], bins=bins) == today_bin]

        # compute stats
        count = len(bucket)
        lo, med, hi = bucket["FutVol"].quantile([0.25, 0.5, 0.75])

        # render as an emphasized markdown block
        st.markdown(
            f"<div style='padding:10px; background-color:#1f2c56; border-radius:5px'>"
            f"<strong style='color:#FFD700'>Empirical outcome:</strong> Over the past "
            f"<strong style='color:#ffffff'>{count}</strong> periods when 7-day "
            f"sentiment MA was ≈ <strong style='color:#ffffff'>{today_val:.2f}</strong>, "
            f"the next <strong style='color:#ffffff'>{horizon}</strong>-day "
            f"realized volatility fell between "
            f"<strong style='color:#00FF7F'>{lo:.2%}</strong> and "
            f"<strong style='color:#FF6347'>{hi:.2%}</strong> (median "
            f"<strong style='color:#ffffff'>{med:.2%}</strong>)." 
            f"</div>",
            unsafe_allow_html=True
        )
