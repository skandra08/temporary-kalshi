"""Speech / press-briefing mention markets: does news attention in the days before an event predict which words get said,
beyond word and speaker base rates, and beyond Kalshi's own price?   python scripts/news_experiment.py
(needs data/mentions_markets.csv and the GDELT cache from digitaledge.news; resumable)"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from digitaledge import news, scoring, screen as S
from digitaledge.http import get_json
from digitaledge.sources.kalshi import BASE

START, END = pd.Timestamp("2026-08-08", tz="UTC"), pd.Timestamp("2026-10-07", tz="UTC")
m = pd.read_csv("data/mentions_markets.csv", usecols=["series", "ticker", "event_ticker", "yes_sub_title", "open_time", "close_time", "result", "volume"])
for c in ("open_time", "close_time"):
    m[c] = pd.to_datetime(m[c], utc=True, format="ISO8601")
m = m[(m.close_time >= START) & (m.close_time < END) & ~m.series.str.startswith("KXEARNINGS") & m.result.isin(["yes", "no"])].copy()
m["label"] = m.yes_sub_title.str.lower().str.strip()
m = m[~m.label.str.contains("does not qualify")]
m["yes"] = (m.result == "yes").astype(int)
m["term"] = m.label.map(news.base_term)
ev = m.groupby("event_ticker").agg(series=("series", "first"), ev_open=("open_time", "min"), settle=("close_time", "max"))
m = m.merge(ev[["ev_open", "settle"]], left_on="event_ticker", right_index=True)

vol = {}
for t in m.term.value_counts()[lambda v: v >= 8].index:
    s = news.daily_volume(t)                      # cached; fetches politely if missing
    if len(s):
        vol[t] = s
m = m[m.term.isin(vol)]
print(f"{len(m)} word-markets, {m.event_ticker.nunique()} events, {m.term.nunique()} terms with news series, {m.series.nunique()} series", flush=True)

# --- causal features
rows, order = [], m.drop_duplicates("event_ticker").sort_values("ev_open")["event_ticker"].tolist()
by_settle = ev.sort_values("settle").index.tolist()
ptr, w_n, w_hit, sw_n, sw_hit, tot_n, tot_hit = 0, {}, {}, {}, {}, 0, 0.0
grp = {e: g for e, g in m.groupby("event_ticker")}
for e in order:
    t0 = ev.loc[e, "ev_open"]
    while ptr < len(by_settle) and ev.loc[by_settle[ptr], "settle"] < t0:
        for r in grp.get(by_settle[ptr], pd.DataFrame()).itertuples():
            w_n[r.label] = w_n.get(r.label, 0) + 1; w_hit[r.label] = w_hit.get(r.label, 0.0) + r.yes
            k = (r.series, r.label); sw_n[k] = sw_n.get(k, 0) + 1; sw_hit[k] = sw_hit.get(k, 0.0) + r.yes
            tot_n += 1; tot_hit += r.yes
        ptr += 1
    gl = (tot_hit + 0.4 * 20) / (tot_n + 20)
    for r in grp[e].itertuples():
        wr = (w_hit.get(r.label, 0.0) + 5 * gl) / (w_n.get(r.label, 0) + 5)
        k = (r.series, r.label)
        swr = (sw_hit.get(k, 0.0) + 3 * wr) / (sw_n.get(k, 0) + 3)
        sal, lev = news.salience(vol[r.term], t0)
        rows.append(dict(ticker=r.ticker, event_ticker=e, series=r.series, label=r.label, yes=r.yes, ev_open=t0, settle=ev.loc[e, "settle"],
                         n_prior=ptr, word_rate=wr, sw_rate=swr, sal=sal, lev=lev, global_rate=gl))
f = pd.DataFrame(rows)
cl = lambda x: np.clip(x, 0.02, 0.98)
f["lw"] = np.log(cl(f.word_rate) / (1 - cl(f.word_rate)))
f["lsw"] = np.log(cl(f.sw_rate) / (1 - cl(f.sw_rate)))
f["sal"] = f["sal"].fillna(0.0); f["lev"] = f["lev"].fillna(f["lev"].median())
f["lev"] = f["lev"] - f["lev"].median()

# --- walk-forward
SETS = {"lr_rates": ["lw", "lsw"], "lr_rates+news": ["lw", "lsw", "sal", "lev"]}
evs = f.drop_duplicates("event_ticker").sort_values("ev_open")
evs = evs[evs.n_prior >= 25]
preds = []
for i in range(0, len(evs), 15):
    blk = evs.iloc[i:i + 15]
    tr = f[f.settle < blk.ev_open.min()]
    te = f[f.event_ticker.isin(blk.event_ticker)].copy()
    if len(tr) < 400 or tr.yes.nunique() < 2:
        continue
    te["p_word"] = te["word_rate"]
    for name, cols in SETS.items():
        te["p_" + name] = LogisticRegression(C=1.0, max_iter=500).fit(tr[cols], tr.yes).predict_proba(te[cols])[:, 1]
    preds.append(te)
w = pd.concat(preds, ignore_index=True)
print(f"\nWalk-forward: {len(w)} word-markets / {w.event_ticker.nunique()} events", flush=True)
rows = []
for mname in ("p_word", "p_lr_rates", "p_lr_rates+news"):
    ll = scoring.logloss(w[mname], w.yes)
    d = scoring.paired_diff(ll, scoring.logloss(w.p_word, w.yes), w.event_ticker, n_boot=2000)
    rows.append(dict(model=mname[2:], logloss=ll.mean(), diff_vs_word=d[0], lo=d[1], hi=d[2]))
print(pd.DataFrame(rows).round(4).to_string(index=False))
d_news = scoring.paired_diff(scoring.logloss(w["p_lr_rates+news"], w.yes), scoring.logloss(w["p_lr_rates"], w.yes), w.event_ticker, n_boot=3000)
print(f"news features' marginal effect (rates+news minus rates): {d_news[0]:+.4f} [{d_news[1]:+.4f}, {d_news[2]:+.4f}]")
w.to_pickle("results/news_walkforward.pkl")

# --- vs the market: price at open+30min (batch 1-min candles), same rule as the pre-registration
mm = w.merge(m[["ticker", "open_time"]].drop_duplicates("ticker"), on="ticker").sort_values("open_time").reset_index(drop=True)
mm["anchor"] = mm.open_time + pd.Timedelta(minutes=30)
groups, cur = [], []
for r in mm.itertuples():
    if cur and (r.open_time - mm.loc[cur[0], "open_time"] > pd.Timedelta(hours=2) or len(cur) >= 40):
        groups.append(cur); cur = []
    cur.append(r.Index)
if cur:
    groups.append(cur)
qrows = []
for g in groups:
    sub = mm.loc[g]
    try:
        dd = get_json(f"{BASE}/markets/candlesticks", {"market_tickers": ",".join(sub.ticker), "period_interval": 1,
                      "start_ts": int(sub.open_time.min().timestamp()), "end_ts": int((sub.open_time.max() + pd.Timedelta(hours=2)).timestamp())})
        qrows += S._parse_batch(dd)
    except Exception:
        continue
c = pd.DataFrame(qrows, columns=["ticker", "ts", "yes_bid", "yes_ask", "volume"]).dropna(subset=["yes_bid", "yes_ask"])
c["t"] = pd.to_datetime(c.ts, unit="s", utc=True)
q = c.merge(mm[["ticker", "anchor"]], on="ticker")
q = q[q.t <= q.anchor].sort_values("t").groupby("ticker").tail(1)
q = q[(q.anchor - q.t) <= pd.Timedelta(minutes=30)]
d = mm.merge(q[["ticker", "yes_bid", "yes_ask"]], on="ticker")
d = d[(d.yes_ask >= d.yes_bid) & (d.yes_ask > 0) & (d.yes_bid < 1) & (d.yes_ask - d.yes_bid <= 0.15)].copy()
d["mid"] = (d.yes_bid + d.yes_ask) / 2
print(f"\nvs market: {len(d)} word-markets with a pre-event quote, {d.event_ticker.nunique()} events", flush=True)
for mname in ("p_word", "p_lr_rates+news"):
    dd_ = scoring.paired_diff(scoring.logloss(d[mname], d.yes), scoring.logloss(d.mid, d.yes), d.event_ticker, n_boot=3000)
    print(f"{mname[2:]:16s} log loss {scoring.logloss(d[mname], d.yes).mean():.4f} vs market {scoring.logloss(d.mid, d.yes).mean():.4f}  diff {dd_[0]:+.4f} [{dd_[1]:+.4f}, {dd_[2]:+.4f}]")
lg = lambda p: np.log(cl(p) / (1 - cl(p)))
d["lm"], d["lp"] = lg(d.mid), lg(d["p_lr_rates+news"])
evd = d.drop_duplicates("event_ticker").sort_values("ev_open")
ps = pd.Series(np.nan, index=d.index)
for i in range(20, len(evd), 10):
    blk = evd.iloc[i:i + 10]
    tr = d[d.settle < blk.ev_open.min()]
    if len(tr) < 150 or tr.yes.nunique() < 2:
        continue
    mdl = LogisticRegression(C=1.0, max_iter=500).fit(tr[["lm", "lp"]], tr.yes)
    idx = d.index[d.event_ticker.isin(blk.event_ticker)]
    ps[idx] = mdl.predict_proba(d.loc[idx, ["lm", "lp"]])[:, 1]
s = d[ps.notna()].assign(p_stack=ps[ps.notna()])
if len(s) > 100:
    dd_ = scoring.paired_diff(scoring.logloss(s.p_stack, s.yes), scoring.logloss(s.mid, s.yes), s.event_ticker, n_boot=3000)
    print(f"stacked(market+model) vs market alone: {scoring.logloss(s.p_stack, s.yes).mean():.4f} vs {scoring.logloss(s.mid, s.yes).mean():.4f}  diff {dd_[0]:+.4f} [{dd_[1]:+.4f}, {dd_[2]:+.4f}] (n={len(s)}, {s.event_ticker.nunique()} events)")
else:
    print("too few rows for a stacking test")
print("\nDONE", flush=True)
