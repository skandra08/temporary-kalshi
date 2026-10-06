"""Data helpers: synthetic price simulation (offline, reproducible) and CSV loading."""
import numpy as np
import pandas as pd


def simulate_prices(n_days=1500, n_assets=5, mu=0.06, vol=0.2, phi=0.0, seed=0):
    """Simulate daily prices. `phi` adds AR(1) autocorrelation to returns
    (phi>0 trending, phi<0 mean-reverting) so strategies have a known edge to detect."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(mu / 252, vol / np.sqrt(252), size=(n_days, n_assets))
    r = np.empty_like(eps)
    r[0] = eps[0]
    for t in range(1, n_days):
        r[t] = eps[t] + phi * (r[t - 1] - mu / 252)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)


def load_csv(path):
    """Load wide-format close prices (date index, one column per asset)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
