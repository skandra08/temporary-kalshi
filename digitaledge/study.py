"""End-to-end study: score the market vs. models on real settled outcomes, test whether the
gap is tradable after fees (with out-of-sample threshold selection), and scan for ladder
arbitrage."""
import json
import numpy as np
import pandas as pd
from .http import utc
from . import dataset, scoring, ladder
from .backtest import trades
from .volmodels import MODELS, predict

UNIVERSE = dict(max_spread=0.10)


def prepare(start, end):
    obs, spot, dv = dataset.build(start, end)
    px = spot.copy()
    px.index = px.index + pd.Timedelta(minutes=1)      # index by candle end time = observation time
    obs = obs[(obs["yes_bid"] >= 0.01) & (obs["yes_ask"] <= 0.99)
              & (obs["yes_ask"] - obs["yes_bid"] <= UNIVERSE["max_spread"])].copy()
    obs["mid"] = (obs["yes_bid"] + obs["yes_ask"]) / 2
    obs = predict(obs, px)
    # evaluate only after the model has enough warm-up history
    first_day = utc(start)
    return obs[obs["obs_time"] >= first_day].reset_index(drop=True), px


def score_table(obs, tau):
    d = obs[obs["tau"] == tau]
    rows = []
    for name, col in [("market_mid", "mid")] + [(m, "p_" + m) for m in MODELS]:
        b, ll = scoring.brier(d[col], d["outcome"]), scoring.logloss(d[col], d["outcome"])
        a, s = scoring.calibration_slope(d[col], d["outcome"])
        db = scoring.paired_diff(b, scoring.brier(d["mid"], d["outcome"]), d["event_ticker"])
        rows.append(dict(tau=tau, model=name, n=len(d), brier=b.mean(), logloss=ll.mean(),
                         cal_intercept=a, cal_slope=s,
                         d_brier_vs_mkt=db[0], ci_lo=db[1], ci_hi=db[2]))
    return pd.DataFrame(rows)


def walk_forward_trading(obs, pcol, edges=(0.0, 0.02, 0.05, 0.08, 0.12), split=0.5, **kw):
    """Pick the edge threshold with the best in-sample mean P&L on the first `split` of
    days, then report P&L of that rule on the later days only."""
    days = np.sort(obs["obs_time"].dt.floor("D").unique())
    cut = days[int(len(days) * split)]
    train, test = obs[obs["obs_time"] < cut], obs[obs["obs_time"] >= cut]
    best, best_pnl = None, -np.inf
    for e in edges:
        t = trades(train, pcol, edge=e, **kw)
        if len(t) >= 30 and t["pnl"].mean() > best_pnl:
            best, best_pnl = e, t["pnl"].mean()
    if best is None:
        return None
    t = trades(test, pcol, edge=best, **kw)
    if t.empty:
        return dict(model=pcol, edge=best, n=0)
    m, lo, hi = scoring.cluster_bootstrap(t["pnl"], t["event_ticker"])
    return dict(model=pcol, edge=best, train_pnl=best_pnl, n=len(t), events=t["event_ticker"].nunique(),
                pnl=m, lo=lo, hi=hi, hit=float((t["pnl"] > 0).mean()))


def run(start, end, out_json=None):
    obs, px = prepare(start, end)
    report = {"n_obs": len(obs), "n_events": int(obs["event_ticker"].nunique()),
              "span": [str(obs["obs_time"].min()), str(obs["obs_time"].max())]}
    report["scores"] = pd.concat([score_table(obs, t) for t in sorted(obs["tau"].unique())]
                                 ).round(5).to_dict("records")
    best_models = ["p_" + m for m in MODELS]
    report["trading"] = [r for m in best_models for fee in ("amortised", "rounded")
                         if (r := walk_forward_trading(obs, m, fee=fee)) and not r.update(fee=fee)]
    lad = ladder.monotonicity_violations(obs)
    report["ladder"] = dict(violations=len(lad), net_positive=int((lad["net"] > 0).sum()) if len(lad) else 0,
                            mean_gross=float(lad["gross"].mean()) if len(lad) else 0.0)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(report, f, indent=1, default=str)
    return obs, report
