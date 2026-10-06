"""Signals use only data up to and including t; the engine lags them one bar before trading."""
import numpy as np
import pandas as pd


def _cs_normalize(s: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally demean and scale to unit gross exposure (dollar-neutral)."""
    s = s.sub(s.mean(axis=1), axis=0)
    gross = s.abs().sum(axis=1).replace(0, np.nan)
    return s.div(gross, axis=0).fillna(0.0)


def momentum(prices, lookback=60, skip=5):
    """Cross-sectional momentum: past `lookback` return, skipping the latest `skip` days."""
    return _cs_normalize(prices.shift(skip) / prices.shift(lookback + skip) - 1)


def mean_reversion(prices, window=20):
    """Cross-sectional mean reversion: short assets stretched above their rolling mean."""
    z = (prices - prices.rolling(window).mean()) / prices.rolling(window).std()
    return _cs_normalize(-z)
