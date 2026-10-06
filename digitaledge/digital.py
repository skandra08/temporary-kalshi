"""Pricing of a digital (cash-or-nothing) contract: P(S_T >= K) given spot and a vol forecast."""
import numpy as np
from scipy import stats

# Kalshi settles on the average of the last 60 one-second index prints. The variance of
# an arithmetic average of Brownian motion over the final window w (minutes) is
# (tau - w) + w/3 instead of tau, so the effective horizon is shorter by 2w/3.
SETTLEMENT_WINDOW_MIN = 1.0


def effective_horizon(tau_min, window=SETTLEMENT_WINDOW_MIN):
    return np.maximum(np.asarray(tau_min, dtype=float) - 2 * window / 3, 1e-6)


def unit_cdf(x, dist="normal", nu=None, z_sorted=None):
    """CDF of a unit-variance standardised shock under the chosen tail model."""
    if dist == "normal":
        return stats.norm.cdf(x)
    if dist == "t":
        scale = np.sqrt((nu - 2) / nu)
        return stats.t.cdf(np.asarray(x) / scale, nu)
    if dist == "emp":   # empirical CDF of past standardised outcomes, lightly smoothed
        n = len(z_sorted)
        return (np.searchsorted(z_sorted, x, side="right") + 0.5) / (n + 1)
    raise ValueError(dist)


def digital_prob(spot, strike, var_per_min, tau_min, dist="normal", nu=None, z_sorted=None,
                 settlement_adjust=True):
    """P(S_T >= K) for log-return r = -v/2 + sqrt(v) Z (martingale, zero drift),
    v = var_per_min * effective horizon."""
    tau = effective_horizon(tau_min) if settlement_adjust else np.asarray(tau_min, dtype=float)
    v = np.asarray(var_per_min) * tau
    sd = np.sqrt(v)
    z_star = (np.log(np.asarray(strike) / np.asarray(spot)) + v / 2) / sd
    return 1.0 - unit_cdf(z_star, dist, nu, z_sorted)


def implied_vol_per_min(spot, strike, prob, tau_min, settlement_adjust=True):
    """Invert the lognormal digital price for sigma per sqrt(minute), given a market probability."""
    from scipy.optimize import brentq
    tau = float(effective_horizon(tau_min)) if settlement_adjust else float(tau_min)
    f = lambda s: digital_prob(spot, strike, s * s, tau, settlement_adjust=False) - prob
    lo, hi = 1e-6, 0.05
    if f(lo) * f(hi) > 0:
        return float("nan")
    return brentq(f, lo, hi)


def kalshi_fee(price, contracts=1, rate=0.07, maker=False):
    """Kalshi fee in dollars: ceil_to_cent(rate * C * P * (1-P)); maker pays ~25% of that."""
    raw = rate * contracts * price * (1 - price) * (0.25 if maker else 1.0)
    return np.ceil(raw * 100 - 1e-9) / 100


def fee_per_contract_amortised(price, rate=0.07, maker=False):
    """Fee when the cent-rounding is amortised over a large order (lower bound on cost)."""
    return rate * price * (1 - price) * (0.25 if maker else 1.0)
