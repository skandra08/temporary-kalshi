"""Kalshi public market data for the hourly BTC above/below ladder (series KXBTCD)."""
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from ..http import get_json, utc

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def settled_markets(start, end, series="KXBTCD", min_volume=1):
    """All settled markets in [start, end) closing time, keeping only strikes that traded."""
    out, cursor = [], None
    s = int(utc(start).timestamp())
    e = int(utc(end).timestamp())
    while True:
        p = {"series_ticker": series, "status": "settled", "limit": 1000,
             "min_close_ts": s, "max_close_ts": e}
        if cursor:
            p["cursor"] = cursor
        d = get_json(f"{BASE}/markets", p)
        out.extend(d["markets"])
        cursor = d.get("cursor")
        if not cursor or not d["markets"]:
            break
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["volume"] = df["volume_fp"].astype(float)
    df = df[(df["volume"] >= min_volume) & df["result"].isin(["yes", "no"])].copy()
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df["strike"] = df["floor_strike"].astype(float)
    df["outcome"] = (df["result"] == "yes").astype(int)
    return df[["ticker", "event_ticker", "strike", "strike_type", "close_time", "outcome",
               "volume"]].reset_index(drop=True)


def _candles(args):
    ticker, series, start_ts, end_ts = args
    d = get_json(f"{BASE}/series/{series}/markets/{ticker}/candlesticks",
                 {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1})
    rows = []
    for c in d.get("candlesticks", []):
        f = lambda x: float(x) if x not in (None, "") else float("nan")
        rows.append((ticker, pd.to_datetime(c["end_period_ts"], unit="s", utc=True),
                     f(c["yes_bid"].get("close_dollars")), f(c["yes_ask"].get("close_dollars")),
                     f(c["price"].get("close_dollars")), f(c["volume_fp"])))
    return rows


def candlesticks(markets, series="KXBTCD", lookback_min=75, workers=3):
    """1-minute bid/ask/last candles for the final `lookback_min` minutes of each market."""
    jobs = [(m.ticker, series, int((m.close_time - pd.Timedelta(minutes=lookback_min)).timestamp()),
             int(m.close_time.timestamp())) for m in markets.itertuples()]
    with ThreadPoolExecutor(workers) as ex:
        rows = [r for chunk in ex.map(_candles, jobs) for r in chunk]
    df = pd.DataFrame(rows, columns=["ticker", "t", "yes_bid", "yes_ask", "last", "volume"])
    return df
