import numpy as np
import pytest
from mmsim import (ASQuoter, FixedSpread, FlowParams, HawkesAwareAS, evaluate,
                   run_strategy, simulate_path)


def test_hawkes_mean_rate_matches_theory():
    p = FlowParams(horizon=2000.0)
    n = np.mean([len(simulate_path(p, s).t) for s in range(8)]) / p.horizon
    assert n == pytest.approx(2 * p.mean_rate_per_side, rel=0.1)


def test_hawkes_clusters_more_than_poisson():
    """Inter-arrival times of a Hawkes process are overdispersed (CV > 1)."""
    path = simulate_path(FlowParams(horizon=3000.0), 1)
    dt = np.diff(path.t)
    assert dt.std() / dt.mean() > 1.1


def test_nonstationary_params_rejected():
    with pytest.raises(ValueError):
        simulate_path(FlowParams(alpha_same=1.0, alpha_cross=0.2), 0)


def test_poisson_fill_rate_matches_analytic():
    """No excitation, no impact: fills per side = mu * P(depth > d) = mu * exp(-k d)."""
    p = FlowParams(alpha_same=0, alpha_cross=0, impact=0, horizon=4000.0)
    d = 1.5
    fills = [run_strategy(simulate_path(p, s), FixedSpread(d), max_inv=10**9).n_fills
             for s in range(6)]
    expected = 2 * p.mu * np.exp(-p.k * d) * p.horizon
    assert np.mean(fills) == pytest.approx(expected, rel=0.1)


def test_no_impact_symmetric_quotes_earn_spread_with_positive_markout():
    p = FlowParams(alpha_same=0, alpha_cross=0, impact=0, sigma=0.05)
    res = evaluate(p, [FixedSpread(2.0)], n_paths=20, max_inv=10**9)["fixed"]
    assert np.mean([r.pnl for r in res]) > 0
    assert np.mean([r.markout for r in res]) > 0


def test_impact_creates_adverse_selection():
    """Same quotes, but with informed flow the markout per fill must be lower."""
    base = dict(alpha_same=0, alpha_cross=0, sigma=0.05)
    mo = {}
    for imp in (0.0, 1.0):
        r = evaluate(FlowParams(impact=imp, **base), [FixedSpread(2.0)], n_paths=20)["fixed"]
        mo[imp] = np.mean([x.markout for x in r])
    assert mo[1.0] < mo[0.0]


def test_inventory_limit_respected():
    path = simulate_path(FlowParams(), 3)
    r = run_strategy(path, FixedSpread(0.1), max_inv=3)
    assert abs(r.inventory_final) <= 3


def test_as_controls_inventory_better_than_fixed():
    p = FlowParams()
    res = evaluate(p, [FixedSpread(2.0), ASQuoter(0.05)], n_paths=40)
    assert (np.mean([r.inventory_std for r in res["avellaneda-stoikov"]])
            < np.mean([r.inventory_std for r in res["fixed"]]))
