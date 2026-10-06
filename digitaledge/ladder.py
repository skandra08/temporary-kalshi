"""Strike-ladder diagnostics. Within one event, P(S>=K) must be non-increasing in K. A
violation (higher strike's bid above a lower strike's ask) is a static arbitrage."""
import numpy as np
import pandas as pd
from .digital import fee_per_contract_amortised


def monotonicity_violations(obs):
    """For each (event, time) find adjacent traded strikes where bid(K_high) > ask(K_low).
    Profit per pair = bid_high - ask_low - fees (buy YES low at ask, buy NO high at 1-bid)."""
    rows = []
    for (ev, t), g in obs.groupby(["event_ticker", "obs_time"]):
        g = g.sort_values("strike")
        lo, hi = g.iloc[:-1], g.iloc[1:]
        gross = hi["yes_bid"].to_numpy() - lo["yes_ask"].to_numpy()
        fees = (fee_per_contract_amortised(lo["yes_ask"].to_numpy())
                + fee_per_contract_amortised(1 - hi["yes_bid"].to_numpy()))
        for i in np.flatnonzero(gross > 0):
            rows.append((ev, t, lo["strike"].iloc[i], hi["strike"].iloc[i], gross[i], gross[i] - fees[i],
                         int(lo["tau"].iloc[i])))
    return pd.DataFrame(rows, columns=["event", "t", "k_low", "k_high", "gross", "net", "tau"])
