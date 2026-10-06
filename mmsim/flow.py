"""Market order flow: bivariate Hawkes process with permanent price impact.

Buys and sells arrive as a self/cross-exciting point process (Ogata thinning). Each
arrival carries an exponential 'depth' (how far through the book it reaches, in ticks)
and moves the mid permanently by `impact` ticks, so flow clusters are informative and
a passive quoter is adversely selected. Prices are in ticks, time in seconds.

The path does not depend on the market maker's quotes, so every strategy can be
evaluated on identical paths (common random numbers) for low-variance comparisons.
"""
from dataclasses import dataclass
import math
import random
import numpy as np


@dataclass(frozen=True)
class FlowParams:
    mu: float = 0.4          # baseline arrival rate per side (events/s)
    alpha_same: float = 0.5  # excitation from same-side event
    alpha_cross: float = 0.1 # excitation from opposite-side event
    beta: float = 1.0        # excitation decay rate (1/s)
    k: float = 0.5           # depth ~ Exp(rate k): mean depth 1/k ticks
    impact: float = 0.4      # permanent mid move per market order (ticks)
    sigma: float = 0.3       # diffusive vol of mid (ticks / sqrt(s))
    horizon: float = 600.0   # seconds

    @property
    def branching_ratio(self):
        """Spectral radius of the kernel matrix; must be < 1 for stationarity."""
        return (self.alpha_same + self.alpha_cross) / self.beta

    @property
    def mean_rate_per_side(self):
        return self.mu / (1 - self.branching_ratio)


@dataclass
class Path:
    t: np.ndarray         # event times
    sign: np.ndarray      # +1 market buy (hits asks), -1 market sell (hits bids)
    depth: np.ndarray     # ticks the order reaches into the book
    mid_pre: np.ndarray   # mid just before the event
    mid_post: np.ndarray  # mid just after (includes impact)
    exc_buy: np.ndarray   # buy excitation just before the event
    exc_sell: np.ndarray
    params: FlowParams

    def mid_at(self, times):
        """Mid-price at arbitrary times (last post-event value; diffusion ignored in
        between, which is fine for horizon-scale markouts)."""
        idx = np.searchsorted(self.t, times, side="right") - 1
        return np.where(idx >= 0, self.mid_post[np.maximum(idx, 0)], 0.0)


def simulate_path(p: FlowParams, seed: int) -> Path:
    if p.branching_ratio >= 1:
        raise ValueError("Hawkes process is non-stationary (branching ratio >= 1)")
    rng = random.Random(seed)
    t, eb, es, mid = 0.0, 0.0, 0.0, 0.0
    T, S, D, MP, MQ, EB, ES = ([] for _ in range(7))
    while True:
        lam_bound = 2 * p.mu + eb + es      # intensity only decays between events
        w = rng.expovariate(lam_bound)
        if t + w > p.horizon:
            break
        decay = math.exp(-p.beta * w)
        eb, es = eb * decay, es * decay
        t += w
        mid += p.sigma * math.sqrt(w) * rng.gauss(0, 1)
        lam_b, lam_s = p.mu + eb, p.mu + es
        u = rng.random() * lam_bound
        if u > lam_b + lam_s:
            continue                          # thinned
        sign = 1 if u < lam_b else -1
        depth = rng.expovariate(p.k)
        T.append(t); S.append(sign); D.append(depth)
        MP.append(mid); EB.append(eb); ES.append(es)
        mid += sign * p.impact
        MQ.append(mid)
        if sign == 1:
            eb += p.alpha_same; es += p.alpha_cross
        else:
            es += p.alpha_same; eb += p.alpha_cross
    a = np.array
    return Path(a(T), a(S), a(D), a(MP), a(MQ), a(EB), a(ES), p)
