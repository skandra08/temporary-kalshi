"""Do the market's own central intervals cover the outcome at their stated rate? Conformal fix + ACI.

python scripts/conformal_study.py --data /path/to/data
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from digitaledge.conformal import (aci, build_ladders, central_score, randomized_pit, split_conformal)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--seeds", type=int, default=50)
a = ap.parse_args()
d = Path(a.data)
markets = pd.read_csv(d / "screen_markets.csv", low_memory=False)
candles = pd.read_csv(d / "screen_candles.csv")
rng = np.random.default_rng(0)


def boot(x, n=2000):
    x = np.asarray(x, float); m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def native_ratio(lads, qhat, alpha):
    """Mean width of the conformal set relative to the market's nominal (1-alpha) set (unit-free)."""
    r = []
    for l, q in zip(lads, qhat):
        wn = l.quantile(1 - alpha / 2) - l.quantile(alpha / 2); wc = l.quantile(.5 + q) - l.quantile(.5 - q)
        if wn > 1e-9 and np.isfinite(wc): r.append(wc / wn)
    return float(np.mean(r))


out = {}
for H in (24, 6, 1):
    L = build_ladders(markets, candles, H)
    n = len(L); cut = int(0.4 * n)
    U = np.stack([randomized_pit(L, seed=s)[0] for s in range(a.seeds)])          # seeds x events
    S = central_score(U)
    res = {"n_events": n, "n_series": len({l.series for l in L})}
    mid = U.mean(0)
    res["pit"] = {"mean_|u-0.5| (0.25 if calibrated)": float(S.mean()),
                  "tail_mass u<.05|u>.95 (0.10 if calibrated)": float(((U < .05) | (U > .95)).mean()),
                  "mean_u (0.5)": float(U.mean())}
    for alpha in (0.2, 0.1, 0.05):
        nominal = (S <= 0.5 - alpha / 2)                                          # market's own central set
        sp_cov, sp_q = [], []
        for s in range(a.seeds):
            q = split_conformal(S[s, :cut], alpha); sp_q.append(q); sp_cov.append(S[s, cut:] <= q)
        sp_cov = np.array(sp_cov).mean(0)
        qhat = np.mean(sp_q)
        acis = np.array([aci(S[s], alpha, gamma=0.02, warm=30)[0] for s in range(a.seeds)]).mean(0)
        r = {"nominal_coverage_test": boot(nominal[:, cut:].mean(0)), "nominal_coverage_all": boot(nominal.mean(0)),
             "split_conformal_coverage_test": boot(sp_cov), "qhat": float(qhat),
             "width_ratio_vs_market": native_ratio(L[cut:], [qhat] * (n - cut), alpha),
             "aci_coverage": boot(1 - acis), "aci_coverage_last40pct": boot(1 - acis[int(.6 * len(acis)):])}
        # per-series conditional coverage of split conformal (series with >= 8 test events)
        ser = np.array([l.series for l in L[cut:]]); per = {}
        for sname in set(ser):
            if (ser == sname).sum() >= 8: per[sname] = float(sp_cov[ser == sname].mean())
        if per: r["series_coverage_min_med_max"] = [min(per.values()), float(np.median(list(per.values()))), max(per.values())]
        res[f"alpha={alpha}"] = r
        print(f"H={H}h n={n} alpha={alpha}: market nominal {1-alpha:.2f} -> actual {r['nominal_coverage_test'][0]:.3f} "
              f"| split-conformal {r['split_conformal_coverage_test'][0]:.3f} (width x{r['width_ratio_vs_market']:.2f}) "
              f"| ACI {r['aci_coverage'][0]:.3f}", flush=True)
    print("  PIT diagnostics:", res["pit"], flush=True)
    out[f"H={H}h"] = res
Path("results").mkdir(exist_ok=True)
Path("results/conformal_results.json").write_text(json.dumps(out, indent=1))
