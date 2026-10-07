"""Live order-book + trade collector for active Kalshi mention markets (data for PREREGISTRATION.md, addendum H2).

    python scripts/book_collector.py --hours 12          # run anywhere (cloud or your laptop)

Writes data_live/book_<UTC stamp>.jsonl, one JSON object per line:
  {"k":"book","ts":...,"ticker":...,"yes_bid":..,"yes_ask":..,"yes":[[price,qty],...],"no":[[price,qty],...]}
  {"k":"trade","ts":...,"ticker":...,"id":...,"yes_price":..,"count":..,"taker_side":...}
Public endpoints only; polite rate (<= ~6 req/s). Stop any time; files are valid after every line."""
import argparse
import json
import os
import time
import pandas as pd
from digitaledge.http import get_json
from digitaledge.sources.kalshi import BASE

DEFAULT_SERIES = ["KXTRUMPMENTION", "KXTRUMPMENTIONB", "KXTRUMPSAY", "KXVANCEMENTION", "KXLASTWORDMENTION",
                  "KXSECPRESSMENTION", "KXFOXNEWSMENTION", "KXWORLDNEWSMENTION", "KXPSAKIMENTION", "KXMAMDANIMENTION"]
ap = argparse.ArgumentParser()
ap.add_argument("--series", nargs="*", default=DEFAULT_SERIES)
ap.add_argument("--hours", type=float, default=12)
ap.add_argument("--every", type=float, default=45)
ap.add_argument("--min-vol24", type=float, default=2000)
ap.add_argument("--max-markets", type=int, default=45)
ap.add_argument("--all-mentions", action="store_true", help="watch every Mentions series in data/series_catalog.csv")
ap.add_argument("--fast-days", type=float, default=0, help="prioritise markets expected to settle within this many days")
ap.add_argument("--levels", type=int, default=8)
a = ap.parse_args()

os.makedirs("data_live", exist_ok=True)
out_path = f"data_live/book_{pd.Timestamp.now(tz='UTC'):%Y%m%d_%H%M}.jsonl"
f = open(out_path, "a", buffering=1)
seen_trades, last_ts, tracked, next_discover = set(), {}, [], 0.0
end = time.time() + a.hours * 3600
print("writing", out_path, flush=True)


def now_iso():
    return pd.Timestamp.now(tz="UTC").isoformat()


def discover():
    rows = []
    series = a.series
    if a.all_mentions:
        cat = pd.read_csv("data/series_catalog.csv")
        series = cat[cat["category"] == "Mentions"]["ticker"].tolist()
    now = pd.Timestamp.now(tz="UTC")
    for s in series:
        try:
            d = get_json(f"{BASE}/markets", {"series_ticker": s, "status": "open", "limit": 300}, cache=False)
        except Exception:
            continue
        for m in d.get("markets", []):
            try:
                v = float(m.get("volume_24h_fp") or 0)
            except ValueError:
                v = 0.0
            if v >= a.min_vol24:
                prio = 1
                if a.fast_days:
                    try:
                        exp = pd.Timestamp(m.get("expected_expiration_time") or m["close_time"])
                        prio = 0 if exp <= now + pd.Timedelta(days=a.fast_days) else 1
                    except (ValueError, KeyError):
                        pass
                rows.append((prio, -v, m["ticker"]))
    rows.sort()
    return [t for _, _, t in rows[:a.max_markets]]


cycles = n_books = n_trades = 0
while time.time() < end:
    t0 = time.time()
    if t0 >= next_discover:
        tracked = discover()
        next_discover = t0 + 900
        print(f"{now_iso()} tracking {len(tracked)} markets", flush=True)
    for tk in tracked:
        try:
            ob = get_json(f"{BASE}/markets/{tk}/orderbook", cache=False)["orderbook_fp"]
            yes = sorted(([float(p), float(q)] for p, q in ob.get("yes_dollars", [])), reverse=True)[:a.levels]
            no = sorted(([float(p), float(q)] for p, q in ob.get("no_dollars", [])), reverse=True)[:a.levels]
            yb = yes[0][0] if yes else None
            ya = round(1 - no[0][0], 4) if no else None          # best YES ask = 1 - best NO bid
            f.write(json.dumps({"k": "book", "ts": now_iso(), "ticker": tk, "yes_bid": yb, "yes_ask": ya, "yes": yes, "no": no}) + "\n")
            n_books += 1
        except Exception:
            pass
        try:
            p = {"ticker": tk, "limit": 1000}
            if tk in last_ts:
                p["min_ts"] = last_ts[tk]
            d = get_json(f"{BASE}/markets/trades", p, cache=False)
            for tr in d.get("trades", []):
                if tr["trade_id"] in seen_trades:
                    continue
                seen_trades.add(tr["trade_id"])
                f.write(json.dumps({"k": "trade", "ts": tr["created_time"], "ticker": tk, "id": tr["trade_id"],
                                    "yes_price": float(tr["yes_price_dollars"]), "count": float(tr["count_fp"]),
                                    "taker_side": tr["taker_side"]}) + "\n")
                n_trades += 1
            if d.get("trades"):
                last_ts[tk] = int(pd.Timestamp(d["trades"][0]["created_time"]).timestamp())      # newest first
        except Exception:
            pass
    cycles += 1
    if cycles % 10 == 0:
        print(f"{now_iso()} cycles={cycles} books={n_books} trades={n_trades}", flush=True)
    time.sleep(max(0.0, a.every - (time.time() - t0)))
print("done", flush=True)
