"""Reproduce the daily-rain study: python scripts/rain_report.py  (needs data/rain_*.csv from the
loader in digitaledge/recurring.py). Writes results/rain_results.json and two figures."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from digitaledge import recurring as R, rain_study as S, scoring, weather as W

m = pd.read_csv("data/rain_markets.csv", parse_dates=["open_time", "close_time", "date"])
cs = pd.read_csv("data/rain_candles.csv", parse_dates=["t"])
e = R.entry_table(m, cs)
e["mid"] = (e.yes_bid + e.yes_ask) / 2
e["pnl_no"] = R.pnl_buy_no(e)
out = {"n_markets": len(m), "n_days": int(m.date.nunique()), "yes_rate": float(m.yes.mean())}

out["naive_no_by_lead"] = S.naive_no_table(e).round(5).to_dict("records")
out["naive_no_by_lead_rounded_fee"] = S.naive_no_table(e, rounded=True).round(5).to_dict("records")
out["calibration_24h"] = S.calibration_by_price(e).round(5).to_dict("records")

days = np.sort(e.date.unique()); cut = days[len(days) // 2]
rows = []
for lead in (6, 12, 18, 24):
    g = e[e.lead_h == lead]
    for thr in (0.03, 0.06, 0.10, 0.20, 0.35, 1.01):
        for part, d in (("train", g[g.date < cut]), ("test", g[g.date >= cut])):
            s = d[d.mid < thr]
            if len(s) >= 30:
                mm, lo, hi = scoring.cluster_bootstrap(s.pnl_no, s.date, n_boot=2000)
                rows.append(dict(lead_h=lead, yes_mid_below=min(thr, 1.0), part=part, n=len(s), pnl=mm, lo=lo, hi=hi))
out["split_date"] = str(cut)[:10]
out["train_test"] = pd.DataFrame(rows).round(5).to_dict("records")

feats = W.build_features(m[m.city.isin(W.CITIES)], dayN=1)
g = S.walk_forward_model(e, feats, lead_h=24)
mv = S.model_vs_market(g)
out["forecast_model"] = {k: {a: ([round(x, 5) for x in b] if isinstance(b, tuple) else round(float(b), 5))
                             for a, b in v.items()} for k, v in mv.items()}
out["forecast_model_trading"] = S.model_trading(g).round(5).to_dict("records")

d = e[e.lead_h == 12].groupby("date").pnl_no.sum()
out["daily_pnl_12h"] = dict(mean=float(d.mean()), sd=float(d.std()), worst=float(d.min()), best=float(d.max()),
                            frac_positive=float((d > 0).mean()), days=int(len(d)))
json.dump(out, open("results/rain_results.json", "w"), indent=1)

# figure 1: calibration (priced vs realised) at 24h lead
c = pd.DataFrame(out["calibration_24h"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].plot(c.mid, c.realised_yes, "o-")
ax[0].set_xlabel("YES price (mid)"); ax[0].set_ylabel("realised rain frequency"); ax[0].set_title("Priced vs realised, 24h before close")
ax[1].bar(range(len(c)), c.no_pnl * 100, yerr=[(c.no_pnl - c.no_lo) * 100, (c.no_hi - c.no_pnl) * 100], capsize=2)
ax[1].set_xticks(range(len(c)), [b.split(",")[1].strip(" ]") for b in c.bucket]); ax[1].axhline(0, color="k", lw=.8)
ax[1].set_xlabel("YES price bucket (upper edge)"); ax[1].set_ylabel("NO P&L, cents/contract"); ax[1].set_title("Buy-NO P&L by YES price (95% day-clustered CI)")
fig.tight_layout(); fig.savefig("results/rain_calibration.png", dpi=130); plt.close(fig)

# figure 2: naive always-NO by lead time
t = pd.DataFrame(out["naive_no_by_lead"]).sort_values("lead_h", ascending=False)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.errorbar(range(len(t)), t.pnl * 100, yerr=[(t.pnl - t.lo) * 100, (t.hi - t.pnl) * 100], fmt="o-", capsize=3)
ax.set_xticks(range(len(t)), [f"{h}h" for h in t.lead_h]); ax.axhline(0, color="k", lw=.8)
ax.set_xlabel("entry time before market close (day starts at 24h)"); ax.set_ylabel("cents / contract")
ax.set_title("'Always buy NO' after fees, by entry time"); fig.tight_layout()
fig.savefig("results/rain_naive_no.png", dpi=130)
print("done")
