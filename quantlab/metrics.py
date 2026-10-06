"""Performance metrics incl. Probabilistic and Deflated Sharpe (Bailey & Lopez de Prado)."""
import math
import numpy as np
import pandas as pd
from statistics import NormalDist

_N = NormalDist()
EULER = 0.5772156649015329


def sharpe(r, periods=252):
    r = pd.Series(r).dropna()
    sd = r.std()
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else 0.0


def max_drawdown(r):
    eq = (1 + pd.Series(r).fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def probabilistic_sharpe(r, benchmark_sr=0.0):
    """P(true per-period Sharpe > benchmark), adjusting for skew and kurtosis."""
    r = pd.Series(r).dropna()
    n, sr = len(r), r.mean() / r.std()
    g3, g4 = r.skew(), r.kurt() + 3
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    return float(_N.cdf((sr - benchmark_sr) * math.sqrt(n - 1) / denom))


def deflated_sharpe(r, n_trials, trial_sr_var):
    """PSR against the expected max Sharpe from `n_trials` independent tries
    (per-period units). Penalises parameter/strategy mining."""
    if n_trials <= 1:
        return probabilistic_sharpe(r, 0.0)
    sd = math.sqrt(trial_sr_var)
    emax = sd * ((1 - EULER) * _N.inv_cdf(1 - 1 / n_trials)
                 + EULER * _N.inv_cdf(1 - 1 / (n_trials * math.e)))
    return probabilistic_sharpe(r, emax)


def summary(r):
    r = pd.Series(r).dropna()
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    mdd = max_drawdown(r)
    return {
        "ann_return": float(ann),
        "ann_vol": float(r.std() * math.sqrt(252)),
        "sharpe": sharpe(r),
        "max_drawdown": mdd,
        "calmar": float(ann / abs(mdd)) if mdd < 0 else float("nan"),
        "psr": probabilistic_sharpe(r),
    }
