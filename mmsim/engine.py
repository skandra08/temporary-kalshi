"""Event-driven market-making simulation on a pre-generated flow path."""
from dataclasses import dataclass
import numpy as np
from .flow import FlowParams, Path, simulate_path


@dataclass
class RunResult:
    pnl: float
    inventory_final: int
    inventory_std: float
    n_fills: int
    markout: float      # mean signed P&L per fill, mid(t+tau) vs fill price (ticks)
    fills: list


def run_strategy(path: Path, strat, max_inv=10, markout_tau=10.0) -> RunResult:
    p = path.params
    cash, q = 0.0, 0
    fills, inv_trace = [], []
    for i in range(len(path.t)):
        t, mid = path.t[i], path.mid_pre[i]
        bid_d, ask_d = strat(t, mid, q, path.exc_buy[i], path.exc_sell[i], p)
        if path.sign[i] == 1 and q > -max_inv and path.depth[i] > ask_d:   # our ask lifted
            px = mid + ask_d
            cash += px; q -= 1; fills.append((t, px, -1))
        elif path.sign[i] == -1 and q < max_inv and path.depth[i] > bid_d:  # our bid hit
            px = mid - bid_d
            cash -= px; q += 1; fills.append((t, px, +1))
        inv_trace.append(q)
    final_mid = path.mid_post[-1] if len(path.t) else 0.0
    pnl = cash + q * final_mid
    if fills:
        ft, fpx, fside = map(np.array, zip(*fills))
        m = path.mid_at(np.minimum(ft + markout_tau, p.horizon))
        markout = float(np.mean(fside * (m - fpx)))
    else:
        markout = 0.0
    return RunResult(pnl, q, float(np.std(inv_trace)) if inv_trace else 0.0,
                     len(fills), markout, fills)


def evaluate(params: FlowParams, strategies, n_paths=200, seed0=0, **kw):
    """Run every strategy on the same n_paths paths. Returns {name: per-path stats}."""
    out = {s.name: [] for s in strategies}
    for k in range(n_paths):
        path = simulate_path(params, seed0 + k)
        for s in strategies:
            out[s.name].append(run_strategy(path, s, **kw))
    return out
