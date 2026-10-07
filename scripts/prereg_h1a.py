"""H1a of PREREGISTRATION.md (prospective): earnings-call words with call date >= 2026-10-08.
Run any time; it only *decides* once >= 300 qualifying contracts have settled (interim runs are descriptive).
    python scripts/prereg_h1a.py"""
import json
import numpy as np
import pandas as pd
import requests
from digitaledge import mentions as M, screen as S
from digitaledge.http import get_json
from digitaledge.sources.kalshi import BASE
from digitaledge.digital import fee_per_contract_amortised as fee_fn

FREEZE = pd.Timestamp("2026-10-08", tz="UTC")
TARGET = 300
ser = [x["ticker"] for x in get_json(f"{BASE}/series", {"category": "Mentions"}, cache=False).get("series", [])
       if x["ticker"].startswith("KXEARNINGSMENTION")]
mk = pd.concat([M.list_full(s) for s in ser], ignore_index=True)       # settled markets only
mk["call_date"] = pd.to_datetime(mk["event_ticker"].str[-7:], format="%y%b%d", utc=True, errors="coerce")
mk = mk[(mk.call_date >= FREEZE) & mk.open_time.notna()].copy()
mk["yes"] = (mk.result == "yes").astype(int)
mk["anchor"] = mk.call_date + pd.Timedelta(hours=8)
print(f"{len(mk)} settled earnings word-markets with call date >= {FREEZE:%Y-%m-%d} ({mk.event_ticker.nunique()} calls)")
if mk.empty:
    raise SystemExit("nothing settled yet")
rows = []
mk = mk.sort_values("open_time")
for i in range(0, len(mk), 40):
    sub = mk.iloc[i:i + 40]
    end = min(sub.anchor.max() + pd.Timedelta(hours=1), pd.Timestamp.now(tz="UTC"))
    try:
        d = get_json(f"{BASE}/markets/candlesticks", {"market_tickers": ",".join(sub.ticker), "period_interval": 60,
                     "start_ts": int(sub.open_time.min().timestamp()), "end_ts": int(end.timestamp())})
    except Exception as e:                      # skip a bad batch rather than abort the whole check
        print("batch skipped:", e)
        continue
    rows += S._parse_batch(d)
c = pd.DataFrame(rows, columns=["ticker", "ts", "yes_bid", "yes_ask", "volume"]).dropna(subset=["yes_bid", "yes_ask"])
c["t"] = pd.to_datetime(c["ts"], unit="s", utc=True)
q = c.merge(mk[["ticker", "anchor"]], on="ticker")
q = q[q.t <= q.anchor].sort_values("t").groupby("ticker").tail(1)
q = q[(q.anchor - q.t) <= pd.Timedelta(hours=12)]
d = mk.merge(q[["ticker", "yes_bid", "yes_ask"]], on="ticker")
d = d[(d.yes_ask >= d.yes_bid) & (d.yes_ask > 0) & (d.yes_bid < 1) & (d.yes_ask - d.yes_bid <= 0.15)].copy()
d["mid"] = (d.yes_bid + d.yes_ask) / 2
t = d[(d.mid > 0.15) & (d.mid <= 0.30)].copy()
print(f"{len(d)} markets with a valid pre-call quote; {len(t)} in the 15-30c bucket from {t.event_ticker.nunique()} calls (decision at {TARGET})")
if len(t) < 5:
    raise SystemExit("too few to report")
ask_no = 1 - t.yes_bid.to_numpy()
t["cost"] = ask_no + fee_fn(ask_no)
t["pnl"] = (1 - t.yes) - t.cost
mean, lo, hi, pb = S.boot_mean(t.pnl, t.event_ticker, n_boot=5000)
pe = S.exact_binom_p(t.pnl.to_numpy(), t.cost.to_numpy(), t.event_ticker.to_numpy())
res = dict(n_contracts=len(t), n_calls=int(t.event_ticker.nunique()), mean_pnl_cents=100 * mean, ci_lo_cents=100 * lo, ci_hi_cents=100 * hi,
           realised_yes=float(t.yes.mean()), mean_yes_mid=float(t.mid.mean()), p_exact_one_sided=pe,
           decision_ready=bool(len(t) >= TARGET), supported=bool(len(t) >= TARGET and pe < 0.025 and lo > 0))
for k, v in res.items():
    print(f"{k:20s} {v:.4f}" if isinstance(v, float) else f"{k:20s} {v}")
json.dump(res, open("results/prereg_h1a_latest.json", "w"), indent=1)
