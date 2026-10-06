"""Walk-forward volatility and tail models.

Everything here is *causal*: forecasts for observation day D use only data whose target
window closed before D 00:00 UTC. Variance units are (log-return)^2 per minute.
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize_scalar
from scipy import stats
from .digital import digital_prob, effective_horizon

MIN_PER_YEAR = 525600.0
GRID_STEP = 15
SEAS_DAYS = 7


class MinuteData:
    """Cumulative sum of squared 1-minute log returns over a complete minute grid."""

    def __init__(self, px: pd.Series):
        self.px = px.astype(float)
        self.t0 = self.px.index[0]
        lp = np.log(self.px.to_numpy())
        r = np.diff(lp, prepend=lp[0])
        self.lp, self.r = lp, r
        self.cs = np.cumsum(r * r)            # cs[i] = sum_{j<=i} r_j^2
        self.n = len(px)

    def pos(self, ts):
        return ((pd.DatetimeIndex(ts) - self.t0) / pd.Timedelta(minutes=1)).astype(int).to_numpy()

    def rv(self, p, w):
        """Mean per-minute variance over the w minutes ending at position p."""
        return (self.cs[p] - self.cs[p - w]) / w

    def features(self, p, tau):
        """HAR-style features at positions p for a target window of `tau` minutes."""
        p = np.asarray(p)
        seas = np.mean([(self.cs[p + tau - k * 1440] - self.cs[p - k * 1440]) / tau
                        for k in range(1, SEAS_DAYS + 1)], axis=0)
        return np.column_stack([np.ones(len(p)), self.rv(p, 60), self.rv(p, 1440),
                                self.rv(p, 10080), seas])

    def ewma_var(self, p, halflife):
        """EWMA of squared returns at positions p (computed once on the whole grid)."""
        key = ("ewma", halflife)
        if not hasattr(self, "_c"):
            self._c = {}
        if key not in self._c:
            self._c[key] = pd.Series(self.r ** 2).ewm(halflife=halflife, adjust=False).mean().to_numpy()
        return self._c[key][np.asarray(p)]

    def horizon_return(self, p, tau):
        return self.lp[np.asarray(p) + tau] - self.lp[np.asarray(p)]


SCALE = 1e8


def fit_har(md: MinuteData, tau, last_pos):
    """Non-negative least squares for per-minute variance over the next `tau` minutes,
    using only grid points whose target window ends at or before `last_pos`."""
    lo = 10080 + 1
    g = np.arange(lo, last_pos - tau + 1, GRID_STEP)
    X = md.features(g, tau) * SCALE
    y = (md.cs[g + tau] - md.cs[g]) / tau * SCALE
    X[:, 0] = 1.0
    coef, _ = nnls(X, y)
    return coef, g


def fit_tail(md, coef, g, tau):
    """Fit a unit-variance Student-t nu and keep sorted standardised residuals for the
    empirical model, from training residuals only."""
    X = md.features(g, tau) * SCALE
    X[:, 0] = 1.0
    vf = np.maximum(X @ coef / SCALE, 1e-12)
    eff = float(effective_horizon(tau))
    z = md.horizon_return(g, tau) / np.sqrt(vf * eff)
    z = z + 0.0
    zs = z / z.std()   # unit variance by construction; absorbs any variance-forecast bias

    def nll(nu):
        s = np.sqrt((nu - 2) / nu)
        return -stats.t.logpdf(zs / s, nu).sum() + len(zs) * np.log(s)
    nu = minimize_scalar(nll, bounds=(2.5, 60), method="bounded").x
    return nu, np.sort(zs), z.std()


MODELS = ["gauss_dvol", "gauss_ewma", "gauss_har", "t_har", "emp_har"]


def predict(obs: pd.DataFrame, px: pd.Series, models=MODELS, ewma_halflife=240):
    """Add columns p_<model> (and var_<model>) to `obs`, fitting walk-forward by UTC day."""
    md = MinuteData(px)
    out = obs.copy().reset_index(drop=True)
    out["pos"] = md.pos(out["obs_time"])
    out["day"] = out["obs_time"].dt.floor("D")
    for m in models:
        out["p_" + m] = np.nan
    spot, K, tau = out["spot"].to_numpy(), out["strike"].to_numpy(), out["tau"].to_numpy()

    if "gauss_dvol" in models:
        v = (out["dvol"].to_numpy() ** 2) / MIN_PER_YEAR
        out["p_gauss_dvol"] = digital_prob(spot, K, v, tau)
    if "gauss_ewma" in models:
        v = md.ewma_var(out["pos"].to_numpy(), ewma_halflife)
        out["p_gauss_ewma"] = digital_prob(spot, K, v, tau)

    har_models = [m for m in models if m.endswith("_har")]
    if har_models:
        for (day, h), idx in out.groupby(["day", "tau"]).groups.items():
            last_pos = int(md.pos([day])[0])      # only targets closed before day start
            coef, g = fit_har(md, int(h), last_pos)
            nu, zs, zsd = fit_tail(md, coef, g, int(h))
            rows = out.loc[idx]
            X = md.features(rows["pos"].to_numpy(), int(h)) * SCALE
            X[:, 0] = 1.0
            v = np.maximum(X @ coef / SCALE, 1e-12) * zsd**2
            args = (rows["spot"].to_numpy(), rows["strike"].to_numpy(), v, rows["tau"].to_numpy())
            if "gauss_har" in models:
                out.loc[idx, "p_gauss_har"] = digital_prob(*args)
            if "t_har" in models:
                out.loc[idx, "p_t_har"] = digital_prob(*args, dist="t", nu=nu)
            if "emp_har" in models:
                out.loc[idx, "p_emp_har"] = digital_prob(*args, dist="emp", z_sorted=zs)
            out.loc[idx, "var_har"] = v
    return out
