"""Analyses for the recurring daily-rain markets. All uncertainty is clustered by calendar day,
because every city shares the same weather systems on a given day."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from . import recurring as R
from . import scoring


def ci(x, days):
    m, lo, hi = scoring.cluster_bootstrap(x, days, n_boot=3000)
    return m, lo, hi


def naive_no_table(e, rounded=False):
    rows = []
    for h, g in e.groupby("lead_h"):
        p = R.pnl_buy_no(g, rounded)
        m, lo, hi = ci(p, g["date"])
        rows.append(dict(lead_h=h, n=len(g), yes_rate=g["yes"].mean(), mean_no_ask=(1 - g["yes_bid"]).mean(),
                         pnl=m, lo=lo, hi=hi, roi=m / (1 - g["yes_bid"]).mean(), win_rate=1 - g["yes"].mean()))
    return pd.DataFrame(rows)


def calibration_by_price(e, lead_h=24, bins=(0, .03, .06, .10, .20, .35, .5, .7, .9, 1.0)):
    g = e[e["lead_h"] == lead_h].copy()
    g["mid"] = (g["yes_bid"] + g["yes_ask"]) / 2
    g["bucket"] = pd.cut(g["mid"], bins, include_lowest=True)
    rows = []
    for b, d in g.groupby("bucket", observed=True):
        no_pnl = R.pnl_buy_no(d)
        yes_pnl = R.pnl_buy_yes(d)
        rows.append(dict(bucket=str(b), n=len(d), mid=d["mid"].mean(), realised_yes=d["yes"].mean(),
                         no_pnl=no_pnl.mean(), no_lo=ci(no_pnl, d["date"])[1], no_hi=ci(no_pnl, d["date"])[2],
                         yes_pnl=yes_pnl.mean()))
    return pd.DataFrame(rows)


def by_city(e, lead_h=24):
    g = e[e["lead_h"] == lead_h]
    rows = []
    for c, d in g.groupby("city"):
        p = R.pnl_buy_no(d)
        rows.append(dict(city=c, n=len(d), yes_rate=d["yes"].mean(), mean_yes_mid=((d.yes_bid + d.yes_ask) / 2).mean(),
                         no_pnl=p.mean(), se=p.std() / np.sqrt(len(d))))
    return pd.DataFrame(rows).sort_values("no_pnl", ascending=False)


def _fit_logit(X, y, l2=1.0):
    def nll(b):
        z = X @ b
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(b[1:] ** 2)
    return minimize(nll, np.zeros(X.shape[1]), method="BFGS").x


def design(d):
    return np.column_stack([np.ones(len(d)), np.log1p(d["fc_mm"]), d["fc_wet_hours"] / 24.0,
                            np.log(np.clip(d["clim"], 0.02, 0.98) / (1 - np.clip(d["clim"], 0.02, 0.98)))])


def walk_forward_model(e, feats, lead_h=24, min_train_days=20):
    """Daily expanding-window logistic model: P(rain) from archived forecast + city climatology
    (past-only base rate). Returns the lead-24h frame with `p_model` and `mid`."""
    g = e[e["lead_h"] == lead_h].merge(feats, on="ticker").dropna(subset=["fc_mm"]).sort_values("date").copy()
    g["mid"] = (g["yes_bid"] + g["yes_ask"]) / 2
    g["clim"] = np.nan
    days = np.sort(g["date"].unique())
    pm = pd.Series(np.nan, index=g.index)
    for i, day in enumerate(days):
        past = g[g["date"] < day]
        cur = g["date"] == day
        base = past.groupby("city")["yes"].agg(["sum", "count"])
        prior = past["yes"].mean() if len(past) else 0.25
        clim = g.loc[cur, "city"].map(lambda c: (base["sum"].get(c, 0) + 3 * prior) / (base["count"].get(c, 0) + 3))
        g.loc[cur, "clim"] = clim.to_numpy()
        if i < min_train_days:
            continue
        tr = g[g["date"] < day].copy()
        tr["clim"] = tr["clim"].fillna(prior)
        b = _fit_logit(design(tr), tr["yes"].to_numpy(float))
        pm[cur] = 1 / (1 + np.exp(-design(g[cur]) @ b))
    g["p_model"] = pm
    return g.dropna(subset=["p_model"])


def model_vs_market(g):
    out = {}
    for name, col in [("market_mid", "mid"), ("model", "p_model"), ("climatology", "clim")]:
        b = scoring.brier(g[col], g["yes"]); ll = scoring.logloss(g[col], g["yes"])
        out[name] = dict(brier=b.mean(), logloss=ll.mean())
    for name, col in [("model", "p_model"), ("climatology", "clim")]:
        out[name]["d_brier_vs_mkt"] = scoring.paired_diff(scoring.brier(g[col], g["yes"]),
                                                         scoring.brier(g["mid"], g["yes"]), g["date"])
    return out


def model_trading(g, edges=(0.0, 0.03, 0.06, 0.10)):
    """Buy whichever side the model says is mispriced by more than `edge` after fees."""
    rows = []
    for e_ in edges:
        yes_ev = g["p_model"] - g["yes_ask"] - R.fee_per_contract_amortised(g["yes_ask"].to_numpy())
        no_ask = 1 - g["yes_bid"]
        no_ev = (1 - g["p_model"]) - no_ask - R.fee_per_contract_amortised(no_ask.to_numpy())
        by, bn = (yes_ev > e_) & (yes_ev >= no_ev), (no_ev > e_) & (no_ev > yes_ev)
        pnl = np.where(by, g["yes"] - g["yes_ask"] - R.fee_per_contract_amortised(g["yes_ask"].to_numpy()),
                       np.where(bn, (1 - g["yes"]) - no_ask - R.fee_per_contract_amortised(no_ask.to_numpy()), np.nan))
        t = g.assign(pnl=pnl, side=np.where(by, "yes", np.where(bn, "no", "")))
        t = t[t["side"] != ""]
        if len(t):
            m, lo, hi = ci(t["pnl"], t["date"])
            rows.append(dict(edge=e_, n=len(t), frac_no=(t["side"] == "no").mean(), pnl=m, lo=lo, hi=hi))
    return pd.DataFrame(rows)
