"""Does knowing WHO bet (Dawid-Skene trader reliability) add information beyond the market price?

Prequential protocol: evidence for each market uses only markets resolved before its anchor time (the K-th bet).
Train the logistic aggregator on the first 60% of anchors whose labels resolved before the test window starts;
test on the last 40%. Null: permute user identities across the sample (keeps vote structure, kills skill).
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from crowdskill.skill import fit_logit, load_votes, predict, replay_features

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, nargs="+", default=[20, 50])
ap.add_argument("--prior", type=float, default=2.0)
a = ap.parse_args()
root = Path("data/manifold")
markets = json.loads((root / "markets.json").read_text())
rng = np.random.default_rng(0)


def brier(p, y): return (p - y) ** 2
def logl(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def boot(d, n=3000):
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


out = {}
for K in a.K:
    recs = load_votes(markets, root / "bets", K)
    n = len(recs); cut = int(.6 * n)
    test_start = recs[cut]["anchor_t"]
    res = {"n_markets": n, "frac_yes": float(np.mean([r["y"] for r in recs]))}
    variants = {"real": replay_features(recs, a.prior)}
    for s in range(3): variants[f"null{s}"] = replay_features(recs, a.prior, shuffle_users=True, seed=s)
    base = variants["real"]
    tr = (np.arange(n) < cut) & (base.resolve_t.to_numpy() < test_start); te = np.arange(n) >= cut
    res["n_train"], res["n_test"] = int(tr.sum()), int(te.sum())
    y = base.y.to_numpy(float)
    w1 = fit_logit(base[["logit_p"]].to_numpy()[tr], y[tr]); p_recal = predict(w1, base[["logit_p"]].to_numpy())
    P = {"market price": base.p.to_numpy(), "recalibrated price": p_recal}
    for name, df in variants.items():
        X = df[["logit_p", "evidence"]].to_numpy(); w = fit_logit(X[tr], y[tr]); P[f"price + evidence ({name})"] = predict(w, X)
        if name == "real": res["coef_logit_p_evidence"] = [float(w[1]), float(w[2])]
    for k, p in P.items():
        res[k] = {"brier": float(brier(p[te], y[te]).mean()), "logloss": float(logl(p[te], y[te]).mean())}
    ref = P["recalibrated price"]
    for k in P:
        if k.startswith("price + evidence"):
            res[k]["brier_diff_vs_recal"] = boot(brier(P[k][te], y[te]) - brier(ref[te], y[te]))
            res[k]["logloss_diff_vs_recal"] = boot(logl(P[k][te], y[te]) - logl(ref[te], y[te]))
    # where the evidence is actually informed: markets with >=5 voters already seen in resolved markets
    inf = te & (base.n_known.to_numpy() >= 5)
    res["n_test_informed(n_known>=5)"] = int(inf.sum())
    if inf.sum() > 50:
        res["informed_logloss_diff_real"] = boot(logl(P["price + evidence (real)"][inf], y[inf]) - logl(ref[inf], y[inf]))
        res["informed_logloss_diff_null0"] = boot(logl(P["price + evidence (null0)"][inf], y[inf]) - logl(ref[inf], y[inf]))
    out[f"K={K}"] = res
    print(f"K={K}  markets={n} train={res['n_train']} test={res['n_test']} informed={res['n_test_informed(n_known>=5)']}")
    for k in P: print(f"  {k:34s} brier {res[k]['brier']:.4f}  logloss {res[k]['logloss']:.4f}")
    print("  logloss diff vs recalibrated price, real :", np.round(res["price + evidence (real)"]["logloss_diff_vs_recal"], 5))
    print("  logloss diff vs recalibrated price, null0:", np.round(res["price + evidence (null0)"]["logloss_diff_vs_recal"], 5))
    print("  coef [logit_p, evidence] =", np.round(res["coef_logit_p_evidence"], 3), flush=True)
Path("results").mkdir(exist_ok=True); Path("results/crowdskill_results.json").write_text(json.dumps(out, indent=1))
