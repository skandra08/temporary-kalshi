"""Descriptive breakdown of settled ghost orders by event and market (how many independent bets, how many went YES).
    python scripts/ghost_outcomes.py [data_dir]"""
import glob, gzip, json, sys
import pandas as pd
from digitaledge.papermaker import simulate_market
from digitaledge.http import get_json
from digitaledge.sources.kalshi import BASE

D = sys.argv[1] if len(sys.argv) > 1 else "data_live"
books, trades = [], []
for fp in sorted(glob.glob(f"{D}/book_*.jsonl")) + sorted(glob.glob(f"{D}/book_*.jsonl.gz")):
    for line in (gzip.open(fp, "rt") if fp.endswith(".gz") else open(fp)):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        (books if r["k"] == "book" else trades).append(r)
b = pd.DataFrame(books); t = pd.DataFrame(trades).drop_duplicates("id")
b["ts"] = pd.to_datetime(b.ts, utc=True, format="ISO8601"); t["ts"] = pd.to_datetime(t.ts, utc=True, format="ISO8601")
g = []
for tk, bb in b.sort_values("ts").groupby("ticker"):
    g += simulate_market(bb, t[t.ticker == tk].sort_values("ts"))
f = pd.DataFrame([vars(x) for x in g]); f = f[f.filled].copy()
f["event"] = f.ticker.str.rsplit("-", n=1).str[0]
res = {}
for tk in f.ticker.unique():
    try:
        res[tk] = get_json(f"{BASE}/markets/{tk}", cache=False)["market"].get("result")
    except Exception:
        res[tk] = None
f["result"] = f.ticker.map(res); d = f[f.result.isin(["yes", "no"])].copy()
d["y"] = (d.result == "yes").astype(int); d["pnl"] = d.ask - d.y
print(f"settled filled ghosts: {len(d)} | markets: {d.ticker.nunique()} | events: {d.event.nunique()}")
print(d.groupby("event").agg(fills=("pnl", "size"), markets=("ticker", "nunique"), yes_markets=("y", lambda x: d.loc[x.index][x == 1].ticker.nunique()),
      avg_ask=("ask", "mean"), pnl_c=("pnl", lambda x: 100 * x.mean())).round(2).to_string())
print(f"\nmarkets that resolved YES: {d[d.y == 1].ticker.nunique()} of {d.ticker.nunique()}; fills on them: {int(d.y.sum())}; mean P&L on those fills: "
      f"{100 * d[d.y == 1].pnl.mean():.1f}c" if d.y.sum() else "\nno filled market has resolved YES yet")
