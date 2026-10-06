"""Assemble the study dataset: one row per (market, observation time) with the quote,
spot at that time, and the realised settlement outcome."""
import pandas as pd
from .http import utc
from .sources import coinbase, kalshi, deribit

HORIZONS = (60, 45, 30, 20, 10, 5)   # minutes before close at which we observe the quote


def build(start, end, cache_csv=None):
    mk = kalshi.settled_markets(start, end)
    cs = kalshi.candlesticks(mk)
    spot = coinbase.close_series(utc(start) - pd.Timedelta(days=30),
                                 utc(end) + pd.Timedelta(hours=1))
    dv = deribit.dvol_hourly(utc(start) - pd.Timedelta(days=1),
                             utc(end) + pd.Timedelta(hours=1))
    rows = []
    close_time = mk.set_index("ticker")["close_time"]
    for h in HORIZONS:
        c = cs.copy()
        c["close_time"] = c["ticker"].map(close_time)
        c["tau"] = ((c["close_time"] - c["t"]).dt.total_seconds() / 60).round().astype(int)
        rows.append(c[c["tau"] == h])
    obs = pd.concat(rows).merge(mk[["ticker", "event_ticker", "strike", "outcome"]], on="ticker")
    # candle ending at t covers [t-1m, t): quote is known at t; spot at that same close time
    obs["obs_time"] = obs["t"]
    obs["spot"] = spot.reindex(obs["obs_time"] - pd.Timedelta(minutes=1)).to_numpy()
    obs["dvol"] = dv.reindex(obs["obs_time"].dt.floor("h"), method="ffill").to_numpy()
    obs = obs.dropna(subset=["spot", "yes_bid", "yes_ask"])
    if cache_csv:
        obs.to_csv(cache_csv, index=False)
    return obs, spot, dv
