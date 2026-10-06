"""Coinbase Exchange BTC-USD 1-minute candles (public, 300 bars per request)."""
import numpy as np
import pandas as pd
from ..http import get_json, utc

URL = "https://api.exchange.coinbase.com/products/{}/candles"


def candles_1m(start, end, product="BTC-USD"):
    """Return a DataFrame indexed by UTC minute-open time with columns open/high/low/close/volume."""
    start, end = utc(start), utc(end)
    rows, cur = [], start
    while cur < end:
        nxt = min(cur + pd.Timedelta(minutes=299), end)
        data = get_json(URL.format(product), {"granularity": 60, "start": cur.isoformat(),
                                              "end": nxt.isoformat()},
                        cache=nxt < pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1))
        rows.extend(data)
        cur = nxt + pd.Timedelta(minutes=1)
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df.drop_duplicates("t").set_index("t").sort_index()


def close_series(start, end):
    """Minute-close price, reindexed to a complete minute grid and forward filled
    (Coinbase omits minutes with no trades)."""
    df = candles_1m(start, end)
    idx = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    return df["close"].astype(float).reindex(idx).ffill()
