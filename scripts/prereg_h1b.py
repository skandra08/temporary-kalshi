"""H1b of PREREGISTRATION.md: buy NO when YES mid in (0.15, 0.30] at open+30min, non-earnings mention markets
closing 2026-08-08 .. 2026-10-06. Run once. Requires data/mentions_markets.csv (see digitaledge.mentions)."""
import json
import numpy as np
import pandas as pd
from digitaledge import mentions as M, screen as S
from digitaledge.http import get_json
from digitaledge.sources.kalshi import BASE
from digitaledge.digital import fee_per_contract_amortised as fee_fn

START, END = pd.Timestamp("2026-08-08", tz="UTC"), pd.Timestamp("2026-10-07", tz="UTC")
allm = pd.read_csv("data/mentions_markets.csv", usecols=["series", "ticker", "event_ticker", "open_time", "close_time", "result", "volume"])
for c in ("open_time", "close_time"):
    allm[c] = pd.to_datetime(allm[c], utc=True, format="ISO8601")
m = allm[~allm.series.str.startswith("KXEARNINGSMENTION") & (allm.close_time >= START) & (allm.close_time < END)
         & allm.result.isin(["yes", "no"]) & allm.open_time.notna()].copy()
m["yes"] = (m.result == "yes").astype(int)
m["anchor"] = m.open_time + pd.Timedelta(minutes=30)
print(f"{len(m)} markets, {m.event_ticker.nunique()} events, {m.series.nunique()} series", flush=True)

# batch 1-minute candles over [open, open+2h], grouping markets that opened close together
m = m.sort_values("open_time").reset_index(drop=True)
groups, cur = [], []
for r in m.itertuples():
    if cur and (r.open_time - m.loc[cur[0], "open_time"] > pd.Timedelta(hours=2) or len(cur) >= 40):
        groups.append(cur); cur = []
    cur.append(r.Index)
if cur:
    groups.append(cur)
rows = []
for i, g in enumerate(groups):
    sub = m.loc[g]
    d = get_json(f"{BASE}/markets/candlesticks", {"market_tickers": ",".join(sub.ticker), "period_interval": 1,
                 "start_ts": int(sub.open_time.min().timestamp()), "end_ts": int((sub.open_time.max() + pd.Timedelta(hours=2)).timestamp())})
    rows += S._parse_batch(d)
    if i % 25 == 0:
        print(f"  batches {i}/{len(groups)}", flush=True)
c = pd.DataFrame(rows, columns=["ticker", "ts", "yes_bid", "yes_ask", "volume"]).dropna(subset=["yes_bid", "yes_ask"])
c["t"] = pd.to_datetime(c["ts"], unit="s", utc=True)
q = c.merge(m[["ticker", "anchor"]], on="ticker")
q = q[q.t <= q.anchor].sort_values("t").groupby("ticker").tail(1)
q = q[(q.anchor - q.t) <= pd.Timedelta(minutes=30)]
d = m.merge(q[["ticker", "yes_bid", "yes_ask"]], on="ticker")
d = d[(d.yes_ask >= d.yes_bid) & (d.yes_ask > 0) & (d.yes_bid < 1)].copy()
d["spread"] = d.yes_ask - d.yes_bid
d = d[d.spread <= 0.15]
d["mid"] = (d.yes_bid + d.yes_ask) / 2
print(f"{len(d)} markets with a valid entry quote ({d.event_ticker.nunique()} events); avg spread {100*d.spread.mean():.1f}c", flush=True)

t = d[(d.mid > 0.15) & (d.mid <= 0.30)].copy()
ask_no = 1 - t.yes_bid.to_numpy()
fee = fee_fn(ask_no)
t["pnl"] = (1 - t.yes) - ask_no - fee
t["cost"] = ask_no + fee
mean, lo, hi, p_boot = S.boot_mean(t.pnl, t.event_ticker, n_boot=5000)
p_exact = S.exact_binom_p(t.pnl.to_numpy(), t.cost.to_numpy(), t.event_ticker.to_numpy())
res = dict(n_contracts=len(t), n_events=int(t.event_ticker.nunique()), mean_pnl_cents=100 * mean, ci_lo_cents=100 * lo,
          ci_hi_cents=100 * hi, win_rate=float((1 - t.yes).mean()), mean_yes_mid=float(t.mid.mean()), realised_yes=float(t.yes.mean()),
          p_boot_one_sided=p_boot, p_exact_one_sided=p_exact, supported=bool(p_exact < 0.025 and lo > 0))
print("\n=== H1b RESULT (pre-registered rule; non-earnings mention markets) ===")
for k, v in res.items():
    print(f"{k:20s} {v:.4f}" if isinstance(v, float) else f"{k:20s} {v}")
json.dump(res, open("results/prereg_h1b.json", "w"), indent=1)
# descriptive only (not part of the decision): same rule across price buckets
d["no_pnl"] = (1 - d.yes) - (1 - d.yes_bid) - fee_fn((1 - d.yes_bid).to_numpy())
d["pb"] = pd.cut(d.mid, [0, .05, .15, .30, .5, .7, .85, .95, 1.0])
print("\n(descriptive, not decisive) buy-NO P&L by YES-price bucket:")
print(d.groupby("pb", observed=True).agg(n=("no_pnl", "size"), mid=("mid", "mean"), realised=("yes", "mean"),
      buyNO_c=("no_pnl", lambda x: 100 * x.mean())).round(2).to_string())
