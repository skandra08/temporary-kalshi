"""Hedging P&L, baselines and the neural hedger, all trained/evaluated on CVaR."""
from __future__ import annotations
import math
import torch
from torch import nn
from .sim import Heston

_N = torch.distributions.Normal(0.0, 1.0)


def payoff(S_T, kind: str, K: float = 1.0):
    return (S_T - K).clamp_min(0) if kind == "call" else (S_T > K).to(S_T.dtype)


def bs_delta(S, tau, sigma, kind: str, K: float = 1.0):
    tau = tau.clamp_min(1e-6)
    sig = sigma.clamp_min(1e-4)
    d1 = (torch.log(S / K) + 0.5 * sig ** 2 * tau) / (sig * tau.sqrt())
    if kind == "call":
        return _N.cdf(d1)
    d2 = d1 - sig * tau.sqrt()
    return _N.log_prob(d2).exp() / (S * sig * tau.sqrt())


def loss_from_policy(S, V, policy, kind: str, cost: float, p: Heston, K: float = 1.0):
    """L = Z - gains + costs (premium omitted: CVaR is translation-invariant).

    policy(t, S_t, V_t, prev_delta) -> delta_t. Costs are proportional to
    |trade| * S, including the final unwind of the position at maturity.
    """
    n, steps = S.shape[0], S.shape[1] - 1
    prev = torch.zeros(n, dtype=S.dtype)
    gains = torch.zeros(n, dtype=S.dtype)
    costs = torch.zeros(n, dtype=S.dtype)
    for t in range(steps):
        d = policy(t, S[:, t], V[:, t], prev)
        costs = costs + cost * (d - prev).abs() * S[:, t]
        gains = gains + d * (S[:, t + 1] - S[:, t])
        prev = d
    costs = costs + cost * prev.abs() * S[:, -1]
    return payoff(S[:, -1], kind, K) - gains + costs


def cvar(L, alpha: float = 0.95):
    q = torch.quantile(L, alpha)
    return (q + (L - q).clamp_min(0).mean() / (1 - alpha)).item()


def delta_policy(p: Heston, kind: str, sigma="const", K: float = 1.0, band: float = 0.0):
    sig0 = math.sqrt(p.theta)

    def pol(t, s, v, prev):
        tau = torch.full_like(s, p.T * (1 - t / p.steps))
        sg = torch.full_like(s, sig0) if sigma == "const" else v.clamp_min(1e-6).sqrt()
        tgt = bs_delta(s, tau, sg, kind, K)
        if band > 0:
            return torch.where((tgt - prev).abs() > band, tgt - band * torch.sign(tgt - prev), prev)
        return tgt
    return pol


class NetHedger(nn.Module):
    """Markov feed-forward hedger; path memory flows through the previous position."""

    def __init__(self, p: Heston, kind: str, use_v: bool = True, width: int = 48, K: float = 1.0):
        super().__init__()
        self.p, self.kind, self.use_v, self.K = p, kind, use_v, K
        nin = 4 + (1 if use_v else 0)
        self.net = nn.Sequential(nn.Linear(nin, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, s, v, prev):
        p = self.p
        tau = torch.full_like(s, p.T * (1 - t / p.steps))
        # residual on BS delta (const vol) -> starts at a sensible hedge, learns the correction
        base = bs_delta(s, tau, torch.full_like(s, math.sqrt(p.theta)), self.kind, self.K)
        scale = 1.0 if self.kind == "call" else 10.0
        feats = [torch.log(s / self.K) * 10, tau * 10, prev / scale, base / scale]
        if self.use_v:
            feats.append((v.clamp_min(0).sqrt() - math.sqrt(p.theta)) * 10)
        x = torch.stack(feats, 1).float()
        return base + scale * self.net(x).squeeze(1).double()


def train(model, p: Heston, kind: str, cost: float, alpha: float = 0.95, iters: int = 400,
          batch: int = 20000, lr: float = 3e-3, seed: int = 0, log=None):
    from .sim import simulate
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    w = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt.add_param_group({"params": [w], "lr": lr * 3})
    for i in range(iters):
        S, V = simulate(p, batch, seed * 100003 + i)
        L = loss_from_policy(S, V, model, kind, cost, p)
        if i == 0:
            w.data.fill_(torch.quantile(L.detach(), alpha))
        obj = w + (L - w).clamp_min(0).mean() / (1 - alpha)
        opt.zero_grad(); obj.backward(); opt.step(); sched.step()
        if log and (i % 50 == 0 or i == iters - 1):
            log(f"  iter {i:4d}  objective {obj.item():.5f}")
    return model
