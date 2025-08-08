#src/models.py

import numpy as np
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense



def fit_garch(returns: np.ndarray, p: int = 1, q: int = 1):
    """
    Fit a GARCH(p,q) model on percent returns (in % units).
    Returns the fitted model object.
    """
    model = arch_model(returns * 100, vol="Garch", p=p, q=q)
    res = model.fit(disp="off")
    return res

def forecast_garch(res, horizon: int = 1):
    """
    Given a fitted GARCH result, produce a volatility forecast.
    """
    f = res.forecast(horizon=horizon)
    # get the forecasted variance for the last available date
    return np.sqrt(f.variance.values[-1]) / 100

def train_rf_multioutput(
    X: np.ndarray,
    Y: np.ndarray,
    **rf_kwargs
) -> MultiOutputRegressor:
    """
    Fit a RandomForest to predict an entire horizon vector at once.
    """
    base = RandomForestRegressor(**rf_kwargs)
    mor = MultiOutputRegressor(base)
    mor.fit(X, Y)
    return mor

def train_nn_multioutput(
    X: np.ndarray,
    Y: np.ndarray,
    hidden_units: int = 64,
    epochs: int = 50,
    batch_size: int = 32
):
    """
    Train a simple feed‑forward NN with horizon outputs.
    """
    model = Sequential([
        Dense(hidden_units, activation="relu", input_shape=(X.shape[1],)),
        Dense(Y.shape[1])  # output dimension = horizon
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def train_xgb_multioutput(
    X: np.ndarray,
    Y: np.ndarray,
    n_estimators: int = 100,
    max_depth:    int = 5,
    learning_rate: float = 0.1,
    subsample:     float = 1.0,
    colsample_bytree: float = 1.0,
    reg_alpha:     float = 0.0,
    reg_lambda:    float = 1.0
) -> MultiOutputRegressor:
    """
    Fit a MultiOutput XGBoost regressor that predicts an entire horizon vector at once.
    """
    base = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective='reg:squarederror',
        verbosity=0
    )
    mor = MultiOutputRegressor(base)
    mor.fit(X, Y)
    return mor