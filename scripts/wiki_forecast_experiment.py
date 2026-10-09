"""DeepAR vs statistical baselines for 28-day-ahead probabilistic forecasts of Wikipedia pageviews.

Train once on data strictly before the first test origin; evaluate at 6 rolling origins (28 days apart) with a frozen model.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, torch
from attention.baselines import QS, empirical_recent, empirical_weekday, lognormal_weekday
from attention.data import fetch_series
from attention.deepar import covariates, train_deepar

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--retrain", action="store_true", help="retrain at every origin on data before it")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_series", type=int, default=700)
a = ap.parse_args()
torch.set_num_threads(4)
CTX, H, NORIG = 90, 28, 6
idx = pd.date_range("2023-01-01", "2026-09-30"); keys = [i.strftime("%Y%m%d") for i in idx]
rows, names = [], []
for f in sorted(Path("data/wiki").glob("*.json"))[:a.max_series * 3]:
    s = json.loads(f.read_text()); v = np.array([s["views"].get(k, np.nan) for k in keys], float)
    if np.isnan(v).mean() > 0.03: continue
    rows.append(np.nan_to_num(v)); names.append(s["article"])
    if len(rows) >= a.max_series: break
Y = np.stack(rows); n, T = Y.shape
print(f"{n} series, {T} days", flush=True)
t_train = T - H * NORIG
t0 = time.time()
model = train_deepar(Y, idx, t_train, ctx=CTX, pred=H, steps=a.steps, seed=a.seed, log=lambda m: print(m, flush=True))
print(f"train {time.time()-t0:.0f}s", flush=True)
model.eval()
covs = torch.from_numpy(covariates(idx, 0, T))

methods = ["DeepAR", "empirical weekday (8 wk)", "log-normal weekday", "empirical last 28d"]
preds = {m: [] for m in methods}; truth = []; ctxs = []
for k in range(NORIG):
    o = t_train + k * H
    if a.retrain and k > 0:
        model = train_deepar(Y, idx, o, ctx=CTX, pred=H, steps=a.steps, seed=a.seed, log=lambda m: None); model.eval()
        print(f"retrained at origin {k}", flush=True)
    ctx = Y[:, o - CTX:o]; truth.append(Y[:, o:o + H]); ctxs.append(ctx)
    cv = covs[o - CTX:o + H][None].repeat(n, 1, 1)
    smp = model.sample(torch.from_numpy(ctx.astype(np.float32)), cv, torch.arange(n), H, n_samples=200, seed=k).numpy()
    preds["DeepAR"].append(np.quantile(smp, QS, axis=1).transpose(1, 0, 2))
    preds["empirical weekday (8 wk)"].append(empirical_weekday(ctx, H))
    preds["log-normal weekday"].append(lognormal_weekday(ctx, H))
    preds["empirical last 28d"].append(empirical_recent(ctx, H))
truth = np.stack(truth, 1)                                 # n, O, H
P = {m: np.stack(v, 1) for m, v in preds.items()}          # n, O, Q, H
rng = np.random.default_rng(0)


def wql(Pm):
    """per-series scale-free weighted quantile loss: 2*sum QL / sum|y| averaged over tau."""
    y = truth[:, :, None, :]; q = QS[None, None, :, None]
    ql = np.maximum(q * (y - Pm), (q - 1) * (y - Pm))
    return 2 * ql.sum((1, 2, 3)) / (len(QS) * np.abs(truth).sum((1, 2)) + 1e-9)


def cover(Pm, lo, hi):
    il, ih = list(QS).index(lo), list(QS).index(hi)
    return ((truth >= Pm[:, :, il]) & (truth <= Pm[:, :, ih])).mean((1, 2))


def boot(x, nb=3000):
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)]
    return [float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


out = {"retrain": a.retrain, "n_series": n, "T": T, "origins": NORIG, "steps": a.steps, "methods": {}}
loss = {m: wql(P[m]) for m in methods}
best = min((m for m in methods if m != "DeepAR"), key=lambda m: loss[m].mean())
spiky = np.std(Y[:, t_train - 90:t_train], 1) / (1 + Y[:, t_train - 90:t_train].mean(1))
terc = np.digitize(spiky, np.quantile(spiky, [1 / 3, 2 / 3]))
for m in methods:
    r = {"wQL": boot(loss[m]), "coverage_90 (nominal .90)": boot(cover(P[m], .05, .95)),
         "coverage_50 (nominal .50)": boot(cover(P[m], .25, .75))}
    if m == "DeepAR":
        r["wQL_diff_vs_best_baseline"] = boot(loss[m] - loss[best]); r["best_baseline"] = best
        r["win_rate_vs_best_baseline"] = float((loss[m] < loss[best]).mean())
    med = P[m][:, :, list(QS).index(0.5)]
    r["median_bias (sum(median-y)/sum(y); 0 is unbiased)"] = boot((med - truth).sum((1, 2)) / (truth.sum((1, 2)) + 1e-9))
    out["methods"][m] = r
    print(f"{m:26s} wQL {r['wQL'][0]:.4f} [{r['wQL'][1]:.4f},{r['wQL'][2]:.4f}]  cov90 {r['coverage_90 (nominal .90)'][0]:.3f}  cov50 {r['coverage_50 (nominal .50)'][0]:.3f}  median bias {r['median_bias (sum(median-y)/sum(y); 0 is unbiased)'][0]:+.3f}")
d = out["methods"]["DeepAR"]["wQL_diff_vs_best_baseline"]
print(f"DeepAR - {best}: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  win rate {out['methods']['DeepAR']['win_rate_vs_best_baseline']:.2f}")
out["by_spikiness_tercile"] = {}
for t, lab in enumerate(["calm", "medium", "spiky"]):
    s = terc == t
    out["by_spikiness_tercile"][lab] = {m: {"wQL": float(loss[m][s].mean()), "cov90": float(cover(P[m], .05, .95)[s].mean())} for m in ("DeepAR", best)}
    print(lab, out["by_spikiness_tercile"][lab])
Path("results").mkdir(exist_ok=True); Path(f"results/wiki_forecast_results_steps{a.steps}{"_retrain" if a.retrain else ""}.json").write_text(json.dumps(out, indent=1))
