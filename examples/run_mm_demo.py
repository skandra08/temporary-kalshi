"""Compare quoting strategies on identical Hawkes order-flow paths, with paired tests."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mmsim import ASQuoter, FixedSpread, FlowParams, HawkesAwareAS, evaluate

N = 400
strats = [FixedSpread(2.0), ASQuoter(0.05), HawkesAwareAS(0.05, widen=0.5, name="hawkes-aware-AS")]
res = evaluate(FlowParams(), strats, n_paths=N, seed0=10_000)  # held-out seeds (widen tuned on 0-299)
pnl = {n: np.array([r.pnl for r in rs]) for n, rs in res.items()}
ref = pnl["avellaneda-stoikov"]

print(f"{'strategy':20s} {'PnL':>8s} {'sd':>7s} {'Sharpe':>7s} {'inv sd':>7s} {'fills':>6s} {'markout':>8s}  paired vs A-S (t)")
for n, rs in res.items():
    x = pnl[n]; d = x - ref
    t = d.mean() / (d.std(ddof=1) / np.sqrt(N)) if d.std() > 0 else float("nan")
    print(f"{n:20s} {x.mean():8.1f} {x.std():7.1f} {x.mean()/x.std():7.2f} "
          f"{np.mean([r.inventory_std for r in rs]):7.2f} {np.mean([r.n_fills for r in rs]):6.0f} "
          f"{np.mean([r.markout for r in rs]):8.2f}  {t:+.2f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for n, x in pnl.items():
    ax[0].hist(x, bins=30, alpha=0.5, label=n)
ax[0].set_title("Terminal P&L over paths (ticks)"); ax[0].legend(fontsize=7)
ax[1].scatter([r.inventory_std for r in res["fixed"]], pnl["fixed"], s=6, label="fixed")
ax[1].scatter([r.inventory_std for r in res["avellaneda-stoikov"]], ref, s=6, label="A-S")
ax[1].set_xlabel("inventory std"); ax[1].set_ylabel("P&L"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("examples/mm_results.png", dpi=120)
