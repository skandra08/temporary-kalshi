"""Apply the frozen H2 rule (PREREGISTRATION.md addendum) to the collected order-book/trade files.
    python scripts/paper_maker.py
Fill statistics are available immediately; P&L needs the markets to have settled (a - y per fill)."""
import glob
import gzip
import json
import sys
import numpy as np
import pandas as pd
import requests
from digitaledge.papermaker import simulate_market
from digitaledge import screen as S
from digitaledge.http import get_json
from digitaledge.sources.kalshi import BASE

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "data_live"      # also reads the archived .jsonl.gz copies
files = sorted(glob.glob(f"{DATA_DIR}/book_*.jsonl")) + sorted(glob.glob(f"{DATA_DIR}/book_*.jsonl.gz"))
books, trades = [], []
for fp in files:
    opener = gzip.open if fp.endswith(".gz") else open
    for line in opener(fp, "rt"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        (books if r["k"] == "book" else trades).append(r)
b = pd.DataFrame(books)
t = pd.DataFrame(trades).drop_duplicates("id")
b["ts"] = pd.to_datetime(b["ts"], utc=True, format="ISO8601"); t["ts"] = pd.to_datetime(t["ts"], utc=True, format="ISO8601")
b = b.sort_values("ts"); t = t.sort_values("ts")
print(f"{len(b)} book snapshots, {len(t)} trades, {b.ticker.nunique()} markets, {b.ts.min():%m-%d %H:%M} -> {b.ts.max():%m-%d %H:%M} UTC")

ghosts = []
for tk, bb in b.groupby("ticker"):
    ghosts += simulate_market(bb, t[t.ticker == tk])
g = pd.DataFrame([vars(x) for x in ghosts])
if g.empty:
    raise SystemExit("no ghost orders (no snapshots with best ask in [3c, 40c])")
g["event"] = g.ticker.str.rsplit("-", n=1).str[0]
print(f"\nghost orders posted {len(g)} | filled {int(g.filled.sum())} ({100*g.filled.mean():.1f}%) | events {g.event.nunique()}")
print(f"median queue ahead at posting: {g.depth_ahead.median():.0f} contracts; filling trades median size {g.fill_trade_size.median():.0f}")

def result(tk):
    try:
        m = get_json(f"{BASE}/markets/{tk}", cache=False)["market"]
        return m.get("result")
    except Exception:
        return None
filled = g[g.filled].copy()
res = {tk: result(tk) for tk in filled.ticker.unique()}
filled["result"] = filled.ticker.map(res)
done = filled[filled.result.isin(["yes", "no"])].copy()
print(f"filled ghosts on settled markets: {len(done)} of {len(filled)} ({done.event.nunique()} events settled)")
if len(done) >= 20:
    done["pnl"] = done.ask - (done.result == "yes").astype(float)
    m, lo, hi, p = S.boot_mean(done.pnl, done.event, n_boot=5000)
    print(f"mean P&L per filled contract: {100*m:+.2f}c  95% CI [{100*lo:+.2f}, {100*hi:+.2f}] (clustered by event)")
    big = done.fill_trade_size >= 20
    for name, d in (("filled by small takers (<20)", done[~big]), ("filled by large takers (>=20)", done[big])):
        if len(d) >= 15:
            mm, l, h, _ = S.boot_mean(d.pnl, d.event, n_boot=3000)
            print(f"  {name:32s} n={len(d):5d}  {100*mm:+.2f}c [{100*l:+.2f}, {100*h:+.2f}]")
    ok = bool(done.event.nunique() >= 30 and lo > 0)
    print(f"\nPRE-REGISTERED DECISION: {'SUPPORTED' if ok else 'NOT supported (needs >= 30 events and CI lower bound > 0)'}; events so far = {done.event.nunique()}")
else:
    print("not enough settled filled orders yet for P&L")
