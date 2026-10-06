"""Demo: does walk-forward momentum survive honest validation? Compare data with a real
edge (trending) vs pure noise, and apply the Deflated Sharpe to account for param mining."""
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from quantlab import backtest, momentum, simulate_prices, walk_forward, summary, deflated_sharpe

GRID = {"lookback": [20, 40, 60, 120], "skip": [0, 5]}
fig, ax = plt.subplots(figsize=(8, 4))
for name, phi in [("trending (phi=0.1)", 0.1), ("noise (phi=0)", 0.0)]:
    p = simulate_prices(3000, 12, mu=0, phi=phi, seed=11)
    oos, log = walk_forward(p, momentum, GRID)
    n = len(list(itertools.product(*GRID.values())))
    trial_var = float(__import__("numpy").var([backtest(p, momentum(p, **dict(zip(GRID, v)))).returns.mean()
                                               / backtest(p, momentum(p, **dict(zip(GRID, v)))).returns.std()
                                               for v in itertools.product(*GRID.values())]))
    s = summary(oos)
    print(f"{name}: OOS Sharpe {s['sharpe']:.2f}  maxDD {s['max_drawdown']:.1%}  "
          f"PSR {s['psr']:.2f}  DSR {deflated_sharpe(oos, n, trial_var):.2f}")
    ((1 + oos).cumprod()).plot(ax=ax, label=name)
ax.set_title("Walk-forward out-of-sample equity"); ax.legend(); fig.tight_layout()
fig.savefig("examples/equity.png", dpi=120)
