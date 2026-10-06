"""As-of weather forecasts for the Kalshi rain cities, from Open-Meteo's previous-runs archive.

`precipitation_previous_dayN` is what the model predicted N days before each valid hour, so
features built from it contain no hindsight. Kalshi settles on the NWS climate day, which is
local *standard* time year-round: the 24h window ending at the market's close_time."""
import numpy as np
import pandas as pd
from .http import get_json

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# NWS climate-station (airport / Central Park) coordinates for Kalshi's KXRAIN cities
CITIES = {
    "Atlanta": (33.64, -84.43), "Austin": (30.19, -97.67), "Boston": (42.36, -71.01),
    "Chicago": (41.98, -87.90), "Houston": (29.65, -95.28), "Denver": (39.86, -104.67),
    "Dallas": (32.90, -97.04), "Miami": (25.79, -80.29), "Los Angeles": (33.94, -118.41),
    "Las Vegas": (36.08, -115.15), "Phoenix": (33.43, -112.01), "Oklahoma City": (35.39, -97.60),
    "New York City": (40.78, -73.97), "New Orleans": (29.99, -90.26), "Minneapolis": (44.88, -93.22),
    "Washington DC": (38.85, -77.04), "Seattle": (47.45, -122.31), "San Antonio": (29.53, -98.47),
    "San Francisco": (37.62, -122.38), "Philadelphia": (39.87, -75.24),
}
VARS = ["precipitation_previous_day1", "precipitation_previous_day2", "precipitation_previous_day3"]


def hourly_forecasts(city, start, end):
    """Hourly archived forecasts (mm) for `city` over [start, end] dates (UTC)."""
    lat, lon = CITIES[city]
    d = get_json(URL, {"latitude": lat, "longitude": lon, "hourly": ",".join(VARS),
                       "start_date": str(start)[:10], "end_date": str(end)[:10], "timezone": "GMT"})
    h = pd.DataFrame(d["hourly"])
    h["time"] = pd.to_datetime(h["time"], utc=True)
    return h.set_index("time")


def window_features(city, hourly, close_time, dayN=1):
    """Features of the forecast for the 24h settlement window ending at close_time."""
    col = f"precipitation_previous_day{dayN}"
    w = hourly.loc[close_time - pd.Timedelta(hours=24) + pd.Timedelta(seconds=1):close_time, col]
    if w.isna().all():
        return dict(fc_mm=np.nan, fc_wet_hours=np.nan, fc_max_mm=np.nan)
    return dict(fc_mm=float(w.sum()), fc_wet_hours=int((w > 0.1).sum()), fc_max_mm=float(w.max()))


def build_features(markets, dayN=1):
    out, cache = [], {}
    for city, g in markets.groupby("city"):
        if city not in CITIES:
            continue
        hourly = hourly_forecasts(city, g["close_time"].min() - pd.Timedelta(days=2), g["close_time"].max())
        for r in g.itertuples():
            f = window_features(city, hourly, r.close_time, dayN)
            out.append(dict(ticker=r.ticker, **f))
    return pd.DataFrame(out)
