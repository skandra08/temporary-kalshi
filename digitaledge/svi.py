"""SVI volatility smile (Gatheral) and the skew-aware digital price.

Total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sig^2)), k = ln(K/F).
A digital call is -dC/dK, which with a smile is  N(d2) - vega * d(sigma)/dK  (r = 0);
the second term is the skew correction that a flat-vol N(d2) misses.
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVI:
    a: float
    b: float
    rho: float
    m: float
    sig: float

    def w(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sig**2))

    def dw(self, k):
        k = np.asarray(k, dtype=float)
        return self.b * (self.rho + (k - self.m) / np.sqrt((k - self.m) ** 2 + self.sig**2))

    def d2w(self, k):
        k = np.asarray(k, dtype=float)
        return self.b * self.sig**2 / ((k - self.m) ** 2 + self.sig**2) ** 1.5


def fit_svi(k, w, weights=None):
    """Least-squares SVI fit to (log-moneyness, total variance) with the usual bounds."""
    k, w = np.asarray(k, float), np.asarray(w, float)
    wt = np.ones_like(w) if weights is None else np.asarray(weights, float)
    best = None
    for rho0 in (-0.4, 0.0, 0.4):
        x0 = [max(w.min() * 0.5, 1e-6), 0.1, rho0, 0.0, 0.1]

        def res(x):
            return (SVI(*x).w(k) - w) * wt
        lo, hi = [-1.0, 1e-6, -0.999, -1.0, 1e-4], [1.0, 10.0, 0.999, 1.0, 2.0]
        r = least_squares(res, x0, bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    return SVI(*best.x)


def call_price(F, K, T, svi: SVI):
    """Undiscounted Black call under the SVI smile."""
    k = np.log(K / F)
    sd = np.sqrt(np.maximum(svi.w(k), 1e-12))
    d1 = (-k + 0.5 * sd**2) / sd
    return F * stats.norm.cdf(d1) - K * stats.norm.cdf(d1 - sd)


def digital_prob_svi(F, K, T, svi: SVI):
    """P(S_T >= K) = N(d2) - vega * dsigma/dK, from the smile (zero rates)."""
    K = np.asarray(K, dtype=float)
    k = np.log(K / F)
    w = np.maximum(svi.w(k), 1e-12)
    sd = np.sqrt(w)
    d1 = (-k + 0.5 * w) / sd
    d2 = d1 - sd
    sigma = sd / np.sqrt(T)
    dsigma_dK = svi.dw(k) / (2 * sigma * T * K)
    vega = F * stats.norm.pdf(d1) * np.sqrt(T)
    return stats.norm.cdf(d2) - vega * dsigma_dK


def density_nonneg(svi: SVI, k=np.linspace(-1.5, 1.5, 601)):
    """Butterfly no-arbitrage condition g(k) >= 0 (Gatheral): risk-neutral density non-negative."""
    w, w1, w2 = svi.w(k), svi.dw(k), svi.d2w(k)
    g = (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2
    return bool(np.all(g >= -1e-9)), float(g.min())


def skew_adjusted_digital(spot, strike, sigma_atm_per_min, tau_min, smile: SVI, tau_smile_min):
    """Level from a realised-vol forecast, shape from the options smile.

    The smile's shape is carried over in standardised moneyness x = k / sqrt(w_atm):
    s(x) = sqrt(w(k)/w_atm). The target-horizon vol is sigma_atm * s(x)."""
    from .digital import effective_horizon
    tau = effective_horizon(tau_min)
    w_atm_t = sigma_atm_per_min**2 * tau
    k_t = np.log(np.asarray(strike) / spot)
    x = k_t / np.sqrt(w_atm_t)
    w_smile_atm = float(smile.w(0.0))                     # ATM-forward total variance of the smile
    root = np.sqrt(w_smile_atm)

    def shape(xx):                                         # relative vol multiplier at std. moneyness
        return np.sqrt(smile.w(xx * root) / w_smile_atm)

    sd = np.sqrt(w_atm_t) * shape(x)
    d1 = (-k_t + 0.5 * sd**2) / sd
    d2 = d1 - sd
    h = 1e-4                                               # d sd / d k = shape'(x), by finite difference
    dsd_dk = (shape(x + h) - shape(x - h)) / (2 * h)
    return stats.norm.cdf(d2) - spot * stats.norm.pdf(d1) * dsd_dk / np.asarray(strike)
