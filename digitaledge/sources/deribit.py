"""Deribit public API: DVOL (30-day implied vol index) and live option book summaries."""
import pandas as pd
from ..http import get_json, utc

BASE = "https://www.deribit.com/api/v2/public"


def dvol_hourly(start, end, currency="BTC"):
    """DVOL close, annualised vol in decimal (0.40 = 40%), indexed by hour (UTC)."""
    s = int(utc(start).timestamp() * 1000)
    e = int(utc(end).timestamp() * 1000)
    rows, cur = [], s
    while cur < e:   # the endpoint returns a limited window per call; page forward
        nxt = min(cur + 30 * 24 * 3600 * 1000, e)
        d = get_json(f"{BASE}/get_volatility_index_data",
                     {"currency": currency, "start_timestamp": cur, "end_timestamp": nxt,
                      "resolution": 3600}, cache=nxt < e - 3600_000)["result"]["data"]
        rows.extend(d)
        cur = nxt + 1
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"]).drop_duplicates("t")
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return (df.set_index("t")["close"] / 100).sort_index()


def option_book(currency="BTC"):
    """Live snapshot of every BTC option: strike, expiry, mark IV, underlying."""
    d = get_json(f"{BASE}/get_book_summary_by_currency", {"currency": currency, "kind": "option"},
                 cache=False)["result"]
    df = pd.DataFrame(d)
    parts = df["instrument_name"].str.split("-", expand=True)
    df["expiry"] = pd.to_datetime(parts[1], format="%d%b%y", utc=True) + pd.Timedelta(hours=8)
    df["strike"] = parts[2].astype(float)
    df["cp"] = parts[3]
    df["iv"] = df["mark_iv"] / 100
    return df[["instrument_name", "expiry", "strike", "cp", "iv", "underlying_price",
               "bid_price", "ask_price", "open_interest"]]
