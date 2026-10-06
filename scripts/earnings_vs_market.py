"""Does the SEC-text model add information beyond Kalshi's own pre-call price, after fees?

Needs results/earnings_walkforward.pkl from scripts/earnings_experiment.py.
Entry price = last quote at or before 08:00 UTC on the call date (before almost every call; the
market lifetime is NOT used because mention markets close early on YES, which would leak)."""
import os
import numpy as np
import pandas as pd
from digitaledge import mentions as M, screen as S, scoring
from digitaledge.digital import fee_per_contract_amortised as fee_fn

w = pd.read_pickle("results/earnings_walkforward.pkl")
w["call_date"] = pd.to_datetime(w["event_ticker"].str[-7:], format="%y%b%d", utc=True)
w["anchor"] = w["call_date"] + pd.Timedelta(hours=8)

# market times for each ticker (cached listing)
ser = sorted({"KXEARNINGSMENTION" + c for c in w["co"].unique()})
mk = pd.concat([M.list_full(s) for s in ser], ignore_index=True)[["series", "ticker", "open_time", "close_time"]]
smp = mk[mk["ticker"].isin(w["ticker"])].drop_duplicates("ticker").copy()
smp["life_h"] = (smp["close_time"] - smp["open_time"]).dt.total_seconds() / 3600
smp = smp[smp["life_h"] > 0.2]
print(f"{len(smp)} markets need quotes", flush=True)
S.fetch_candles_batch(smp, "data/earn_candles.csv", batch=40, workers=2, fallback=True)

c = pd.read_csv("data/earn_candles.csv").dropna(subset=["yes_bid", "yes_ask"])
c = c[c["ts"] > 0]
c["t"] = pd.to_datetime(c["ts"], unit="s", utc=True)
x = w.merge(smp[["ticker", "open_time"]], on="ticker")
x = x[x["anchor"] > x["open_time"] + pd.Timedelta(hours=2)]
q = c.merge(x[["ticker", "anchor"]], on="ticker")
q = q[q["t"] <= q["anchor"]].sort_values("t").groupby("ticker").tail(1)
q = q[(q["anchor"] - q["t"]) <= pd.Timedelta(hours=12)]
d = x.merge(q[["ticker", "yes_bid", "yes_ask"]], on="ticker")
d = d[(d["yes_ask"] > 0) & (d["yes_bid"] < 1) & (d["yes_ask"] >= d["yes_bid"])].copy()
d["mid"] = (d["yes_bid"] + d["yes_ask"]) / 2
d["spread"] = d["yes_ask"] - d["yes_bid"]
d = d[d["spread"] <= 0.15].sort_values("ev_open").reset_index(drop=True)
print(f"{len(d)} word-markets with a clean pre-call quote, {d.event_ticker.nunique()} calls, avg spread {100*d.spread.mean():.1f}c", flush=True)
d.to_pickle("results/earnings_vs_market.pkl")

logit = lambda p: np.log(np.clip(p, 0.02, 0.98) / (1 - np.clip(p, 0.02, 0.98)))
print("\n== 1. Accuracy: model vs market (negative = model better), paired, clustered by call")
for name, col in [("sec-text LR", "p_lr_prior+text"), ("gbm", "p_gbm"), ("per-word rate", "p_word")]:
    for metric, fn in (("logloss", scoring.logloss), ("brier", scoring.brier)):
        dd = scoring.paired_diff(fn(d[col], d.yes), fn(d["mid"], d.yes), d.event_ticker, n_boot=3000)
        print(f"{name:14s} {metric:8s} model {fn(d[col], d.yes).mean():.4f} market {fn(d['mid'], d.yes).mean():.4f}  diff {dd[0]:+.4f} [{dd[1]:+.4f}, {dd[2]:+.4f}]")

print("\n== 2. Does the model add information beyond the market? (walk-forward stacking)")
from sklearn.linear_model import LogisticRegression
d["lm"], d["lp"] = logit(d["mid"]), logit(d["p_lr_prior+text"])
evs = d.drop_duplicates("event_ticker").sort_values("ev_open")
pred = pd.Series(np.nan, index=d.index)
for i in range(30, len(evs), 15):
    blk = evs.iloc[i:i + 15]
    tr = d[d["settle_time"] < blk["ev_open"].min()]
    if len(tr) < 400:
        continue
    m = LogisticRegression(C=1.0, max_iter=500).fit(tr[["lm", "lp"]], tr["yes"])
    idx = d.index[d["event_ticker"].isin(blk["event_ticker"])]
    pred[idx] = m.predict_proba(d.loc[idx, ["lm", "lp"]])[:, 1]
s = d[pred.notna()].copy(); s["p_stack"] = pred[pred.notna()]
full = LogisticRegression(C=1.0, max_iter=500).fit(d[["lm", "lp"]], d["yes"])
print(f"in-sample coefficients: market {full.coef_[0][0]:.3f}, model {full.coef_[0][1]:.3f} (a model coefficient > 0 means it adds information)")
for metric, fn in (("logloss", scoring.logloss), ("brier", scoring.brier)):
    dd = scoring.paired_diff(fn(s["p_stack"], s.yes), fn(s["mid"], s.yes), s.event_ticker, n_boot=3000)
    print(f"stacked vs market-alone {metric:8s}: {fn(s['p_stack'], s.yes).mean():.4f} vs {fn(s['mid'], s.yes).mean():.4f}  diff {dd[0]:+.4f} [{dd[1]:+.4f}, {dd[2]:+.4f}]  (n={len(s)} markets, {s.event_ticker.nunique()} calls)")

print("\n== 3. Trading (taker, fee-aware). Threshold chosen on the first half of calls, evaluated on the second half")
def trades(df, pcol, edge):
    ay, an = df["yes_ask"].to_numpy(), 1 - df["yes_bid"].to_numpy()
    p = df[pcol].to_numpy()
    ey, en = p - ay - fee_fn(ay), (1 - p) - an - fee_fn(an)
    by, bn = (ey > edge) & (ey >= en), (en > edge) & (en > ey)
    y = df["yes"].to_numpy()
    pnl = np.where(by, y - ay - fee_fn(ay), np.where(bn, (1 - y) - an - fee_fn(an), np.nan))
    cost = np.where(by, ay + fee_fn(ay), an + fee_fn(an))
    return df.assign(pnl=pnl, cost=cost, side=np.where(by, "yes", np.where(bn, "no", "")))[lambda z: z.side != ""]
for pcol in ("p_lr_prior+text", "p_stack"):
    base = s if pcol == "p_stack" else d
    cut = base["ev_open"].sort_values().iloc[len(base) // 2]
    tr_, te_ = base[base["ev_open"] < cut], base[base["ev_open"] >= cut]
    best, bp = None, -9
    for e in (0.0, 0.03, 0.05, 0.08, 0.12):
        t = trades(tr_, pcol, e)
        if len(t) >= 40 and t.pnl.mean() > bp:
            best, bp = e, t.pnl.mean()
    if best is None:
        print(pcol, "no threshold with >= 40 training trades"); continue
    t = trades(te_, pcol, best)
    if len(t) < 20:
        print(pcol, f"edge {best}: only {len(t)} test trades"); continue
    m, lo, hi, pb = S.boot_mean(t.pnl, t.event_ticker, n_boot=3000)
    pe = S.exact_binom_p(t.pnl, t.cost, t.event_ticker)
    print(f"{pcol:16s} edge>{best:.2f}: train {100*bp:+.2f}c | TEST n={len(t)} calls={t.event_ticker.nunique()} "
          f"mean {100*m:+.2f}c [{100*lo:+.2f}, {100*hi:+.2f}] hit {100*(t.pnl>0).mean():.0f}% exact-p {pe:.3f}")
print("\nDONE", flush=True)
