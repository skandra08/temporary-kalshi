"""Heston paths (full-truncation Euler, exact-ish with many substeps) in torch."""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Heston:
    s0: float = 1.0
    v0: float = 0.04
    kappa: float = 1.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    T: float = 30 / 365
    steps: int = 30
    sub: int = 4  # Euler substeps per hedge interval


def simulate(p: Heston, n: int, seed: int):
    """Return S, V of shape (n, steps+1). Zero interest rate."""
    g = torch.Generator().manual_seed(seed)
    dt = p.T / (p.steps * p.sub)
    s = torch.full((n,), p.s0, dtype=torch.float64)
    v = torch.full((n,), p.v0, dtype=torch.float64)
    S, V = [s.clone()], [v.clone()]
    sq = math.sqrt(1 - p.rho ** 2)
    for _ in range(p.steps):
        for _ in range(p.sub):
            z1 = torch.randn(n, generator=g, dtype=torch.float64)
            z2 = p.rho * z1 + sq * torch.randn(n, generator=g, dtype=torch.float64)
            vp = v.clamp_min(0)
            s = s * torch.exp(-0.5 * vp * dt + vp.sqrt() * math.sqrt(dt) * z1)
            v = v + p.kappa * (p.theta - vp) * dt + p.xi * vp.sqrt() * math.sqrt(dt) * z2
        S.append(s.clone()); V.append(v.clone())
    return torch.stack(S, 1), torch.stack(V, 1)
