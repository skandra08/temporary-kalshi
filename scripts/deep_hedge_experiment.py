"""Deep hedging vs Black-Scholes delta (+ no-trade band) under Heston and proportional costs.

Train on seeds disjoint from the test seed. Report CVaR95 / mean / std of L = Z - gains + costs
on a fresh 200k-path test set, with a paired bootstrap CI on the CVaR difference to the best
baseline (baselines' band width is tuned on the training distribution, not on test).
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from deephedge.sim import Heston, simulate
from deephedge.hedge import (NetHedger, cvar, delta_policy, loss_from_policy, train)

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="call")
ap.add_argument("--cost", type=float, default=0.005)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ntest", type=int, default=200000)
a = ap.parse_args()
torch.manual_seed(a.seed); torch.set_num_threads(4)
p = Heston()
log = lambda m: print(m, flush=True)

St, Vt = simulate(p, a.ntest, 10_000_000 + a.seed)          # test
Sv, Vv = simulate(p, 50000, 20_000_000 + a.seed)            # tuning for baseline band
res = {}
def ev(name, pol):
    with torch.no_grad():
        L = loss_from_policy(St, Vt, pol, a.kind, a.cost, p)
    res[name] = dict(cvar95=cvar(L), mean=L.mean().item(), std=L.std().item(), L=L)
    log(f"{name:28s} CVaR95 {res[name]['cvar95']:.5f}  mean {res[name]['mean']:.5f}  std {res[name]['std']:.5f}")

best_band, best_c = 0.0, 1e9
for b in [0, .01, .02, .03, .05, .07, .1, .15]:
    with torch.no_grad():
        c = cvar(loss_from_policy(Sv, Vv, delta_policy(p, a.kind, band=b), a.kind, a.cost, p))
    if c < best_c: best_band, best_c = b, c
log(f"[{a.kind}] cost={a.cost} tuned band={best_band}")
ev("BS delta (const vol)", delta_policy(p, a.kind))
ev("BS delta (spot vol)", delta_policy(p, a.kind, sigma="spot"))
ev(f"BS delta + band {best_band}", delta_policy(p, a.kind, band=best_band))
t0 = time.time()
for use_v, name in [(False, "deep hedger (S only)"), (True, "deep hedger (S and v)")]:
    m = train(NetHedger(p, a.kind, use_v=use_v), p, a.kind, a.cost, iters=a.iters, seed=a.seed + 1, log=log)
    ev(name, m)
log(f"train time {time.time()-t0:.0f}s")

base = min((k for k in res if k.startswith("BS")), key=lambda k: res[k]["cvar95"])
g = torch.Generator().manual_seed(1)
n = a.ntest
out = {"kind": a.kind, "cost": a.cost, "best_baseline": base, "band": best_band}
for k in res:
    if not k.startswith("deep"): continue
    diffs = []
    for _ in range(200):
        idx = torch.randint(0, n, (n,), generator=g)
        diffs.append(cvar(res[k]["L"][idx]) - cvar(res[base]["L"][idx]))
    d = torch.tensor(diffs)
    out[k] = dict(cvar_diff_vs_best_baseline=d.mean().item(),
                  ci95=[d.quantile(.025).item(), d.quantile(.975).item()])
    log(f"{k} - {base}: {d.mean():+.5f}  95% CI [{d.quantile(.025):+.5f}, {d.quantile(.975):+.5f}] (paired bootstrap over paths)")
out["table"] = {k: {x: v[x] for x in ("cvar95", "mean", "std")} for k, v in res.items()}
Path("results").mkdir(exist_ok=True)
Path(f"results/deep_hedge_{a.kind}_c{a.cost}_s{a.seed}.json").write_text(json.dumps(out, indent=1))
