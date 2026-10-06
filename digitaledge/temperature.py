"""Daily-high-temperature ladders (Kalshi KXHIGH*): price every bracket from an as-of forecast
plus a walk-forward error model, and compare with the market.

NWS climate reports whole degrees F, so the continuous temperature boundaries sit at +-0.5:
  'between a and b'  (a..b inclusive)  -> [a - 0.5, b + 0.5)
  'greater x'  ('x+1 or above')        -> [x + 0.5, inf)
  'less    y'  ('y-1 or below')        -> (-inf, y - 0.5)
"""
import numpy as np
import pandas as pd
from scipy import stats

from .http import get_json

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def bracket_bounds(strike_type, floor_strike, cap_strike):
    if strike_type == "between":
        return floor_strike - 0.5, cap_strike + 0.5
    if strike_type == "greater":
        return floor_strike + 0.5, np.inf
    if strike_type == "less":
        return -np.inf, cap_strike - 0.5
    raise ValueError(strike_type)


def bracket_prob(mu, sigma, lo, hi, df=None):
    """P(lo <= T < hi) for T ~ Normal(mu, sigma) (or Student-t with `df` degrees of freedom)."""
    cdf = (lambda x: stats.norm.cdf(x)) if df is None else (lambda x: stats.t.cdf(x, df))
    return cdf((hi - mu) / sigma) - cdf((lo - mu) / sigma)


def observed_temp(event_markets):
    """Integer high implied by which bracket resolved YES (midpoint); NaN if a tail bracket won
    (censored) or the event is unresolved."""
    w = event_markets[event_markets["result"] == "yes"]
    if len(w) != 1 or w.iloc[0]["strike_type"] != "between":
        return np.nan
    return (w.iloc[0]["floor_strike"] + w.iloc[0]["cap_strike"]) / 2


def forecast_high_f(lat, lon, start, end, dayN=1):
    """Hourly archived forecast temperature (F) from `precipitation`-style previous-runs API."""
    d = get_json(URL, {"latitude": lat, "longitude": lon, "hourly": f"temperature_2m_previous_day{dayN}",
                       "start_date": str(start)[:10], "end_date": str(end)[:10], "timezone": "GMT",
                       "temperature_unit": "fahrenheit"})
    h = pd.DataFrame(d["hourly"])
    h["time"] = pd.to_datetime(h["time"], utc=True)
    return h.set_index("time")[f"temperature_2m_previous_day{dayN}"]


def window_max(hourly, close_time):
    """Forecast max over the 24h climate day ending at close_time."""
    w = hourly.loc[close_time - pd.Timedelta(hours=24) + pd.Timedelta(seconds=1):close_time]
    return float(w.max()) if len(w) and not w.isna().all() else np.nan


def walk_forward_error_model(days_df, window=30, min_obs=10, sigma_floor=1.5):
    """days_df: one row per (city, date) with `fc` (forecast high) and `obs` (observed, may be NaN).
    For each row use only strictly earlier days of the same city: bias = mean(obs-fc),
    sigma = sd of the debiased errors. Returns the frame with `mu` and `sigma`."""
    out = []
    for city, g in days_df.sort_values("date").groupby("city"):
        g = g.copy()
        err = (g["obs"] - g["fc"])
        mu, sg = [], []
        for i in range(len(g)):
            past = err.iloc[max(0, i - window):i].dropna()
            if len(past) >= min_obs:
                b = past.mean()
                s = max(past.std(ddof=1), sigma_floor)
            else:
                b, s = 0.0, 3.5          # weakly informative prior until enough history
            mu.append(g["fc"].iloc[i] + b)
            sg.append(s)
        g["mu"], g["sigma"] = mu, sg
        out.append(g)
    return pd.concat(out)
