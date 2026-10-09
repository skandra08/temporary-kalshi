import torch
from deephedge.sim import Heston, simulate
from deephedge.hedge import bs_delta, cvar, delta_policy, loss_from_policy, NetHedger


def test_heston_martingale_and_positive():
    S, V = simulate(Heston(), 40000, 1)
    assert (S > 0).all()
    assert abs(S[:, -1].mean().item() - 1.0) < 0.01


def test_bs_delta_matches_autograd():
    from torch.distributions import Normal
    s = torch.tensor([0.95, 1.0, 1.05], dtype=torch.float64, requires_grad=True)
    tau = torch.full((3,), 0.08, dtype=torch.float64); sig = torch.full((3,), 0.2, dtype=torch.float64)
    d1 = (torch.log(s) + 0.5 * sig ** 2 * tau) / (sig * tau.sqrt())
    price = s * Normal(0., 1.).cdf(d1) - Normal(0., 1.).cdf(d1 - sig * tau.sqrt())
    price.sum().backward()
    assert torch.allclose(s.grad, bs_delta(s.detach(), tau, sig, "call"), atol=1e-8)


def test_zero_cost_delta_hedge_beats_unhedged():
    p = Heston(); S, V = simulate(p, 20000, 2)
    hedged = cvar(loss_from_policy(S, V, delta_policy(p, "call"), "call", 0.0, p))
    none = cvar(loss_from_policy(S, V, lambda t, s, v, prev: torch.zeros_like(s), "call", 0.0, p))
    assert hedged < 0.5 * none


def test_net_starts_at_bs_delta_and_costs_penalise_trading():
    p = Heston(); S, V = simulate(p, 5000, 3); m = NetHedger(p, "call")
    a = loss_from_policy(S, V, m, "call", 0.0, p).mean()
    b = loss_from_policy(S, V, delta_policy(p, "call"), "call", 0.0, p).mean()
    assert torch.allclose(a, b, atol=1e-6)
    assert loss_from_policy(S, V, m, "call", 0.01, p).mean() > a
