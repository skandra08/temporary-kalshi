"""Train an RL market maker (linear Q-learning, tile coding) on the Hawkes-flow simulator and benchmark it against
Avellaneda-Stoikov on held-out paths using common random numbers.
    python scripts/rl_mm_experiment.py [--episodes 3000]"""
import argparse
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mmsim import ASQuoter, FixedSpread, FlowParams, HawkesAwareAS, run_strategy, simulate_path
from mmsim.rl import QAgent, RLQuoter, run_episode

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=3000)
ap.add_argument("--eta", type=float, default=0.5)
ap.add_argument("--test-paths", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

base = FlowParams()
ood = FlowParams(mu=0.3, alpha_same=0.6, alpha_cross=0.1, impact=0.8, sigma=0.45)       # stronger clustering, impact and vol
agent = QAgent(seed=a.seed)
val_paths = [simulate_path(base, 50_000 + k) for k in range(40)]
curve = []


def val_mean():
    return float(np.mean([run_episode(p, agent, eps=0.0, learn=False, eta=a.eta)[0] for p in val_paths]))


t0 = time.time()
curve.append((0, val_mean()))
for ep in range(1, a.episodes + 1):
    eps = max(0.02, 0.3 * (1 - ep / (0.7 * a.episodes)))
    run_episode(simulate_path(base, ep), agent, eps=eps, learn=True, eta=a.eta)
    if ep % max(1, a.episodes // 10) == 0:
        curve.append((ep, val_mean()))
        print(f"episode {ep:6d}  validation mean P&L {curve[-1][1]:8.1f} ticks   ({time.time() - t0:.0f}s)", flush=True)


def evaluate(params, n, seed0, label):
    strategies = [FixedSpread(2.0), ASQuoter(0.05), HawkesAwareAS(0.05, widen=0.5, name="hawkes-aware-AS"), RLQuoter(agent)]
    res = {s.name: [run_strategy(simulate_path(params, seed0 + k), s) for k in range(n)] for s in strategies}
    pnl = {k: np.array([r.pnl for r in v]) for k, v in res.items()}
    ref = pnl["avellaneda-stoikov"]
    out = {}
    print(f"\n=== {label}: {n} held-out paths (identical for every strategy) ===")
    print(f"{'strategy':20s} {'P&L':>8s} {'sd':>7s} {'Sharpe':>7s} {'inv sd':>7s} {'fills':>6s}  paired vs A-S: mean diff [95% CI]")
    for k, v in res.items():
        x = pnl[k]; d = x - ref
        boot = np.random.default_rng(0).choice(d, (4000, len(d))).mean(1)
        lo, hi = np.quantile(boot, [.025, .975])
        out[k] = dict(pnl=float(x.mean()), sd=float(x.std()), sharpe=float(x.mean() / x.std()), inv_sd=float(np.mean([r.inventory_std for r in v])),
                      fills=float(np.mean([r.n_fills for r in v])), diff_vs_as=float(d.mean()), diff_lo=float(lo), diff_hi=float(hi))
        print(f"{k:20s} {x.mean():8.1f} {x.std():7.1f} {x.mean() / x.std():7.2f} {out[k]['inv_sd']:7.2f} {out[k]['fills']:6.0f}  {d.mean():+8.1f} [{lo:+.1f}, {hi:+.1f}]")
    return out


results = {"episodes": a.episodes, "eta": a.eta, "curve": curve,
           "in_distribution": evaluate(base, a.test_paths, 100_000, "IN-DISTRIBUTION (same flow as training)"),
           "out_of_distribution": evaluate(ood, a.test_paths, 200_000, "OUT-OF-DISTRIBUTION (stronger clustering, impact, volatility)")}
json.dump(results, open("results/rl_mm_results.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(*zip(*curve), "o-"); ax.axhline(results["in_distribution"]["avellaneda-stoikov"]["pnl"], color="k", ls="--", lw=.8, label="Avellaneda-Stoikov (test)")
ax.set_xlabel("training episodes"); ax.set_ylabel("validation P&L (ticks)"); ax.legend(); ax.set_title("RL market maker learning curve")
fig.tight_layout(); fig.savefig("results/rl_mm_learning_curve.png", dpi=130)
print("\nDONE", flush=True)
