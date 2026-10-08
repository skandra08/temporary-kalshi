"""Exploratory: can a model tell toxic maker fills from benign ones, using only information visible before the trade?

A maker who sells YES at price p to a taker earns p - y (y = 1 if the market resolves YES). We predict y for every taker-YES trade in
the ghost-order price band [0.03, 0.40] from strictly past market activity, with 5-fold CV grouped by EVENT, and compare the realised
maker P&L of the trades the model would have accepted (expected P&L p - p_hat > 0) against all trades.
    python scripts/toxicity_model.py          (needs data/tape_mentions.pkl)"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from digitaledge import screen as S

tr = pd.read_pickle("data/tape_mentions.pkl").sort_values(["ticker", "t"]).reset_index(drop=True)
tr["w"] = tr["count"]
feats = []
for tk, g in tr.groupby("ticker", sort=False):
    t = g["t"].astype("int64").to_numpy() / 1e9
    p, c, side = g["yes_price"].to_numpy(), g["count"].to_numpy(), (g["taker_side"] == "yes").to_numpy()
    n = len(g)
    cumv = np.concatenate([[0.0], np.cumsum(c)[:-1]])
    ret10 = np.zeros(n); imb10 = np.zeros(n); vol10 = np.zeros(n)
    lo = 0
    for i in range(n):
        while t[lo] < t[i] - 600:
            lo += 1
        if lo < i:                                   # strictly earlier trades in the last 10 minutes
            vv = c[lo:i]; pp = p[lo:i]; ss = side[lo:i]
            ret10[i] = p[i] - np.average(pp, weights=vv + 1e-9)
            imb10[i] = (vv[ss].sum() - vv[~ss].sum()) / (vv.sum() + 1e-9)
            vol10[i] = vv.sum()
    feats.append(pd.DataFrame(dict(idx=g.index, ret10=ret10, imb10=imb10, lvol10=np.log1p(vol10), lcum=np.log1p(cumv),
                                   age_h=(t - t[0]) / 3600)))
F = pd.concat(feats).set_index("idx")
tr = tr.join(F)
tr["lsize"] = np.log1p(tr["count"])
d = tr[(tr.taker_side == "yes") & (tr.yes_price >= 0.03) & (tr.yes_price <= 0.40)].copy()
d["pnl"] = d.yes_price - d.yes                         # maker P&L per contract when selling YES at the trade price
cols = ["yes_price", "lsize", "ret10", "imb10", "lvol10", "lcum", "age_h"]
print(f"{len(d)} taker-YES trades in [3c, 40c], {int(d.w.sum())} contracts, {d.event_ticker.nunique()} events; YES rate {np.average(d.yes, weights=d.w):.3f}")

oof = np.zeros(len(d))
for trn, te in GroupKFold(5).split(d, groups=d.event_ticker):
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=2.0, random_state=0)
    m.fit(d.iloc[trn][cols], d.iloc[trn]["yes"], sample_weight=d.iloc[trn]["w"].clip(upper=200))
    oof[te] = m.predict_proba(d.iloc[te][cols])[:, 1]
d["p_hat"] = oof
print(f"out-of-fold AUC for predicting YES: price alone {roc_auc_score(d.yes, d.yes_price, sample_weight=d.w):.3f} | model {roc_auc_score(d.yes, d.p_hat, sample_weight=d.w):.3f}")

def wboot(df, B=3000, seed=0):
    ev = df.groupby("event_ticker").apply(lambda x: pd.Series({"num": (x.pnl * x.w).sum(), "den": x.w.sum()}))
    num, den = ev.num.to_numpy(), ev.den.to_numpy()
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(ev), (B, len(ev)))
    b = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), np.quantile(b, .025), np.quantile(b, .975), len(ev)

d["ev"] = d.yes_price - d.p_hat                           # model's expected maker P&L for this fill
rows = []
for name, sel in (("all fills", d.ev > -9), ("accepted: expected P&L > 0", d.ev > 0), ("accepted: expected P&L > 2c", d.ev > 0.02),
                  ("rejected: expected P&L <= 0", d.ev <= 0)):
    s = d[sel]
    if len(s) > 100:
        m, lo, hi, n = wboot(s, B=2000)
        rows.append(dict(set=name, trades=len(s), contracts=int(s.w.sum()), events=n, maker_pnl_c=100 * m, lo=100 * lo, hi=100 * hi))
print(pd.DataFrame(rows).round(2).to_string(index=False))
big = d.w >= 20
print("\nsame comparison, small-trade fills only (<20 contracts):")
for name, sel in (("all", d.ev > -9), ("accepted ev>0", d.ev > 0)):
    s = d[sel & ~big]
    if len(s) > 100:
        m, lo, hi, n = wboot(s, B=2000); print(f"  {name:14s} trades {len(s):7d}  {100*m:+.2f}c [{100*lo:+.2f}, {100*hi:+.2f}]")
d.to_pickle("results/toxicity_oof.pkl")
