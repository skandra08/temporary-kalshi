"""Live snapshot: compare open Kalshi hourly BTC contracts to (a) flat-vol and (b) options-surface
(skew-aware) fair values, and append to a CSV so forward, out-of-sample evidence accumulates.

    python -m digitaledge.live            # one snapshot; run on a schedule to build a dataset
"""
import os
import sys
import numpy as np
import pandas as pd
from .digital import digital_prob
from .http import get_json
from .sources import coinbase, deribit
from .sources.kalshi import BASE
from .svi import density_nonneg, fit_svi, skew_adjusted_digital

OUT = "results/live_snapshots.csv"


def smile_from_deribit(book, now):
    """Fit SVI to the nearest expiry (>= 6h out) using OTM options; returns (svi, F, T_years, expiry)."""
    b = book[book["expiry"] > now + pd.Timedelta(hours=6)]
    expiry = b["expiry"].min()
    e = b[b["expiry"] == expiry].copy()
    F = float(e["underlying_price"].median())
    T = (expiry - now).total_seconds() / (365 * 86400)
    e["k"] = np.log(e["strike"] / F)
    e = e[((e["cp"] == "P") & (e["strike"] <= F)) | ((e["cp"] == "C") & (e["strike"] >= F))]
    e = e[(e["iv"] > 0) & (e["k"].abs() < 0.15)]
    svi = fit_svi(e["k"].to_numpy(), (e["iv"] ** 2 * T).to_numpy())
    return svi, F, T, expiry, len(e)


def open_markets(series="KXBTCD"):
    d = get_json(f"{BASE}/markets", {"series_ticker": series, "status": "open", "limit": 1000}, cache=False)
    m = pd.DataFrame(d["markets"])
    for c in ("yes_bid_dollars", "yes_ask_dollars"):
        m[c] = m[c].astype(float)
    m["strike"] = m["floor_strike"].astype(float)
    m["close_time"] = pd.to_datetime(m["close_time"], utc=True)
    return m[(m["yes_bid_dollars"] > 0) & (m["yes_ask_dollars"] < 1)]


def snapshot():
    now = pd.Timestamp.now(tz="UTC")
    px = coinbase.close_series(now - pd.Timedelta(days=3), now)
    spot = float(px.iloc[-1])
    r = np.diff(np.log(px.to_numpy()))
    var_min = float(pd.Series(r**2).ewm(halflife=240, adjust=False).mean().iloc[-1])
    book = deribit.option_book()
    svi, F, T, expiry, n_opt = smile_from_deribit(book, now)
    ok, gmin = density_nonneg(svi)
    m = open_markets()
    m["tau"] = (m["close_time"] - now).dt.total_seconds() / 60
    m = m[m["tau"] > 2].copy()
    if m.empty:
        print("no open markets with quotes"); return None
    m["mid"] = (m["yes_bid_dollars"] + m["yes_ask_dollars"]) / 2
    m["p_flat"] = digital_prob(spot, m["strike"], var_min, m["tau"])
    m["p_skew"] = skew_adjusted_digital(spot, m["strike"].to_numpy(), np.sqrt(var_min),
                                        m["tau"].to_numpy(), svi, 0.0)
    m["snap_time"], m["spot"] = now, spot
    keep = m[["snap_time", "ticker", "close_time", "strike", "tau", "spot", "yes_bid_dollars",
              "yes_ask_dollars", "mid", "p_flat", "p_skew"]]
    os.makedirs("results", exist_ok=True)
    keep.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print(f"{now:%H:%M}Z spot {spot:,.0f} | Deribit expiry {expiry:%d%b %H:%M} ({n_opt} opts) "
          f"SVI rho={svi.rho:+.2f} ATM IV={np.sqrt(svi.w(0)/T):.1%} no-arb={ok} | EWMA vol={np.sqrt(var_min*525600):.1%}")
    print(keep.assign(edge_flat=lambda d: d.p_flat - d["mid"], edge_skew=lambda d: d.p_skew - d["mid"])
          .drop(columns=["snap_time", "spot", "close_time"]).round(3).to_string(index=False))
    return keep


if __name__ == "__main__":
    snapshot()
