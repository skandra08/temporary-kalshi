"""Quoting strategies. Each returns (bid_dist, ask_dist): ticks below/above the mid.

State passed in: time t, mid, inventory q, buy/sell Hawkes excitation (observable
in practice as exponentially weighted recent flow counts).
"""
from dataclasses import dataclass
import math


@dataclass
class FixedSpread:
    half_spread: float = 2.0
    name: str = "fixed"

    def __call__(self, t, mid, q, exc_buy, exc_sell, p):
        return self.half_spread, self.half_spread


@dataclass
class ASQuoter:
    """Avellaneda-Stoikov (2008): reservation price shifted against inventory,
    optimal spread from risk aversion gamma and book liquidity k."""
    gamma: float = 0.05
    name: str = "avellaneda-stoikov"

    def reservation_and_spread(self, t, q, p):
        tau = max(p.horizon - t, 1.0)
        var = p.sigma ** 2
        shift = q * self.gamma * var * tau
        spread = self.gamma * var * tau + (2 / self.gamma) * math.log(1 + self.gamma / p.k)
        return shift, spread

    def __call__(self, t, mid, q, exc_buy, exc_sell, p):
        shift, spread = self.reservation_and_spread(t, q, p)
        return spread / 2 + shift, spread / 2 - shift


@dataclass
class HawkesAwareAS(ASQuoter):
    """A-S plus flow-intensity awareness. After a burst of buys the ask is likely to be
    picked off before the price catches up, so widen/skew away from the hot side in
    proportion to its excess intensity (which is observable)."""
    widen: float = 1.0
    name: str = "hawkes-aware-AS"

    def __call__(self, t, mid, q, exc_buy, exc_sell, p):
        bid, ask = super().__call__(t, mid, q, exc_buy, exc_sell, p)
        return bid + self.widen * exc_sell, ask + self.widen * exc_buy
