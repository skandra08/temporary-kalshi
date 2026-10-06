"""Forward-data collector. Samples the open Kalshi BTC ladder, spot, a vol forecast and the
Deribit SVI smile every `--every` seconds for `--minutes` minutes, appending to a CSV.

Designed to be fired hourly by a scheduler. Outcomes are NOT stored: settlement is joined
later from Kalshi, so nothing here can leak the future.

    python -m digitaledge.collect --minutes 55 --every 300
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from .live import open_markets, smile_from_deribit
from .digital import digital_prob
from .sources import coinbase, deribit
from .svi import skew_adjusted_digital

OUT = "data_live/fwd_{stamp}.csv.gz"   # one small file per run: no merge conflicts
MAX_TAU = 90.0           # minutes ahead of settlement
MAX_MONEYNESS = 0.02     # keep strikes within +-2% of spot


def sample(px_hist=None):
    now = pd.Timestamp.now(tz="UTC")
    px = coinbase.close_series(now - pd.Timedelta(days=3), now)
    spot = float(px.iloc[-1])
    r = np.diff(np.log(px.to_numpy()))
    var_ewma = float(pd.Series(r**2).ewm(halflife=240, adjust=False).mean().iloc[-1])
    try:
        svi, F, T, expiry, n_opt = smile_from_deribit(deribit.option_book(), now)
        smile = dict(svi_a=svi.a, svi_b=svi.b, svi_rho=svi.rho, svi_m=svi.m, svi_sig=svi.sig,
                     svi_T=T, deribit_expiry=expiry, n_opt=n_opt, deribit_F=F)
    except Exception:                      # smile is optional; keep collecting the rest
        svi, smile = None, {}
    m = open_markets()
    m["tau"] = (m["close_time"] - now).dt.total_seconds() / 60
    m = m[(m["tau"] > 2) & (m["tau"] <= MAX_TAU)
          & (np.abs(np.log(m["strike"] / spot)) < MAX_MONEYNESS)].copy()
    if m.empty:
        return None
    m["snap_time"], m["spot"], m["var_ewma"] = now, spot, var_ewma
    m["mid"] = (m["yes_bid_dollars"] + m["yes_ask_dollars"]) / 2
    m["p_flat"] = digital_prob(spot, m["strike"], var_ewma, m["tau"])
    if svi is not None:
        m["p_skew"] = skew_adjusted_digital(spot, m["strike"].to_numpy(), np.sqrt(var_ewma),
                                            m["tau"].to_numpy(), svi, 0.0)
    for k, v in smile.items():
        m[k] = v
    cols = ["snap_time", "ticker", "event_ticker", "strike", "close_time", "tau", "spot", "var_ewma",
            "yes_bid_dollars", "yes_ask_dollars", "mid", "p_flat", "p_skew", *smile]
    return m[[c for c in cols if c in m.columns]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=55)
    ap.add_argument("--every", type=float, default=300)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    a.out = a.out.format(stamp=pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    end, n = time.time() + a.minutes * 60, 0
    while time.time() < end:
        t0 = time.time()
        try:
            df = sample()
            if df is not None:
                df.to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
                n += len(df)
        except Exception as e:                # network blips must not kill a long run
            print("sample failed:", repr(e)[:120])
        time.sleep(max(0.0, a.every - (time.time() - t0)))
    print(f"collected {n} rows -> {a.out}")


if __name__ == "__main__":
    main()
