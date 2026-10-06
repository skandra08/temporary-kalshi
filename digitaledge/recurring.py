"""Recurring-market study: does a simple rule (e.g. always buy NO on 'will it rain?') beat fees?

Data: Kalshi KXRAIN, one daily binary market per city ('total precipitation strictly > 0 in').
Entry prices come from 1-hour candles at fixed lead times before close, using only quotes
that existed at that moment; outcomes are the settled results."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from .http import get_json
from .sources.kalshi import BASE
from .digital import fee_per_contract_amortised, kalshi_fee

OFFSETS_H = (42, 36, 30, 24, 18, 12, 6, 3, 1)   # hours before close (24 ~ start of the local day)


def settled_markets(series="KXRAIN"):
    rows = []
    for path in ("markets", "historical/markets"):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000}
            if path == "markets":
                p["status"] = "settled"
            if cur:
                p["cursor"] = cur
            d = get_json(f"{BASE}/{path}", p, cache=True)
            rows += [dict(m, src=path) for m in d.get("markets", [])]
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df = df[df["result"].isin(["yes", "no"])].copy()
    for c in ("open_time", "close_time"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["city"] = df["yes_sub_title"]
    df["yes"] = (df["result"] == "yes").astype(int)
    df["volume"] = df["volume_fp"].astype(float)
    df["date"] = df["close_time"].dt.floor("D")
    return df[["ticker", "event_ticker", "city", "open_time", "close_time", "date", "yes", "volume", "src"]]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _px(block):
    """Close price from a bid/ask block under either key spelling (dollars string or cents)."""
    if not isinstance(block, dict):
        return np.nan
    if block.get("close_dollars") not in (None, ""):
        return _num(block["close_dollars"])
    c = _num(block.get("close"))
    return c / 100 if c == c and c > 1 else c


def _candles(args):
    ticker, series, start, end, src = args
    paths = [f"{BASE}/series/{series}/markets/{ticker}/candlesticks",
             f"{BASE}/historical/markets/{ticker}/candlesticks"]
    if src == "historical/markets":
        paths = paths[::-1]
    for url in paths:
        try:
            d = get_json(url, {"start_ts": int(start.timestamp()), "end_ts": int(end.timestamp()),
                               "period_interval": 60})
        except Exception:
            continue
        cs = d.get("candlesticks", [])
        if cs:
            rows = []
            for c in cs:
                try:
                    rows.append((ticker, pd.to_datetime(c["end_period_ts"], unit="s", utc=True),
                                 _px(c.get("yes_bid")), _px(c.get("yes_ask")), _num(c.get("volume_fp", c.get("volume")))))
                except (KeyError, TypeError, ValueError):
                    continue          # malformed candle: skip it, keep the market
            if rows:
                return rows
    return []


def candles(markets, series="KXRAIN", workers=3):
    jobs = [(m.ticker, series, m.open_time, m.close_time, m.src) for m in markets.itertuples()]
    with ThreadPoolExecutor(workers) as ex:
        rows = [r for chunk in ex.map(_candles, jobs) for r in chunk]
    return pd.DataFrame(rows, columns=["ticker", "t", "yes_bid", "yes_ask", "volume"])


def entry_table(markets, cs, offsets=OFFSETS_H):
    """One row per (market, lead time): the last quote at or before close - offset."""
    cs = cs.sort_values("t")
    out = []
    ct = markets.set_index("ticker")["close_time"]
    for h in offsets:
        q = cs.copy()
        q["cutoff"] = q["ticker"].map(ct) - pd.Timedelta(hours=h)
        q = q[q["t"] <= q["cutoff"]]
        last = q.groupby("ticker").tail(1).assign(lead_h=h)
        out.append(last)
    e = pd.concat(out).merge(markets, on="ticker")
    e = e.dropna(subset=["yes_bid", "yes_ask"])
    e["stale_h"] = (e["close_time"] - pd.to_timedelta(e["lead_h"], unit="h") - e["t"]).dt.total_seconds() / 3600
    return e[e["stale_h"] <= 3].reset_index(drop=True)      # drop quotes older than 3 hours


def pnl_buy_no(e, rounded=False):
    """P&L of buying one NO at the ask (= 1 - yes_bid), paying the taker fee."""
    ask_no = 1 - e["yes_bid"]
    fee = kalshi_fee(ask_no.to_numpy()) if rounded else fee_per_contract_amortised(ask_no.to_numpy())
    return (1 - e["yes"]) - ask_no - fee


def pnl_buy_yes(e, rounded=False):
    ask = e["yes_ask"]
    fee = kalshi_fee(ask.to_numpy()) if rounded else fee_per_contract_amortised(ask.to_numpy())
    return e["yes"] - ask - fee
