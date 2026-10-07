"""Kalshi trade tape. Trading is zero-sum before fees, so on any trade the maker's gross P&L per
contract is exactly the negative of the taker's: measuring takers measures the whole maker side."""
import numpy as np
import pandas as pd
from .http import get_json
from .sources.kalshi import BASE


def trades(ticker, max_pages=8):
    rows, cur = [], None
    for _ in range(max_pages):
        p = {"ticker": ticker, "limit": 1000}
        if cur:
            p["cursor"] = cur
        try:
            d = get_json(f"{BASE}/markets/trades", p)
        except Exception:
            try:
                d = get_json(f"{BASE}/historical/trades", p)
            except Exception:
                break
        rows += d.get("trades", [])
        cur = d.get("cursor")
        if not cur or not d.get("trades"):
            break
    if not rows:
        return pd.DataFrame(columns=["ticker", "t", "yes_price", "count", "taker_side"])
    df = pd.DataFrame(rows)
    return pd.DataFrame(dict(ticker=ticker, t=pd.to_datetime(df["created_time"], utc=True, format="ISO8601"),
                             yes_price=pd.to_numeric(df["yes_price_dollars"]), count=pd.to_numeric(df["count_fp"]),
                             taker_side=df["taker_side"]))


def taker_gross(df, yes):
    """Per-trade taker gross P&L per contract: buys YES at p -> y - p; buys NO at 1-p -> p - y."""
    y = np.asarray(yes, float)
    p = df["yes_price"].to_numpy()
    return np.where(df["taker_side"].to_numpy() == "yes", y - p, p - y)
