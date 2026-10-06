"""Proper scoring rules, calibration, and cluster-robust inference.

Contracts at different strikes in the same hourly event resolve off one BTC print, so their
outcomes are strongly dependent. All uncertainty is therefore computed by resampling
*events* (clusters), never individual contracts."""
import numpy as np
import pandas as pd

EPS = 1e-4


def brier(p, y):
    return (np.asarray(p) - np.asarray(y)) ** 2


def logloss(p, y):
    p = np.clip(np.asarray(p), EPS, 1 - EPS)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def cluster_bootstrap(values, clusters, stat=np.mean, n_boot=2000, seed=0, ci=0.95):
    """Bootstrap CI of stat(values) resampling whole clusters. Returns (point, lo, hi)."""
    v = pd.Series(np.asarray(values, dtype=float))
    g = pd.Series(np.asarray(clusters))
    sums = v.groupby(g).agg(["sum", "count"])
    s, c = sums["sum"].to_numpy(), sums["count"].to_numpy()
    rng = np.random.default_rng(seed)
    k = len(s)
    idx = rng.integers(0, k, size=(n_boot, k))
    boot = s[idx].sum(axis=1) / c[idx].sum(axis=1)       # cluster-resampled mean
    a = (1 - ci) / 2
    return float(s.sum() / c.sum()), float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def paired_diff(score_a, score_b, clusters, **kw):
    """CI for mean(score_a - score_b); negative means a is better for loss-type scores."""
    return cluster_bootstrap(np.asarray(score_a) - np.asarray(score_b), clusters, **kw)


def reliability(p, y, bins=10):
    """Binned reliability table: mean forecast vs realised frequency."""
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    t = df.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return t.reset_index(drop=True)


def calibration_slope(p, y):
    """Logistic recalibration y ~ a + b*logit(p). b=1,a=0 means perfectly calibrated;
    b<1 means forecasts are over-dispersed (too extreme)."""
    from scipy.optimize import minimize
    p = np.clip(np.asarray(p), EPS, 1 - EPS)
    x, y = np.log(p / (1 - p)), np.asarray(y, dtype=float)

    def nll(th):
        z = th[0] + th[1] * x
        return np.sum(np.logaddexp(0, z) - y * z)
    r = minimize(nll, [0.0, 1.0], method="BFGS")
    return float(r.x[0]), float(r.x[1])
