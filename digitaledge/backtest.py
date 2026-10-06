"""Fee-aware trading rule: take liquidity when model value beats the ask by a margin."""
import numpy as np
import pandas as pd
from .digital import fee_per_contract_amortised, kalshi_fee


def trades(df, pcol, edge=0.0, fee="amortised", maker=False):
    """One contract per signal. Buys YES at the ask if p - ask > edge (after fees),
    buys NO at (1 - bid) if (1-p) - (1-bid) > edge. Returns a frame of realised P&L."""
    feefn = (lambda x: fee_per_contract_amortised(x, maker=maker)) if fee == "amortised" \
        else (lambda x: kalshi_fee(x, 1, maker=maker))
    p = df[pcol].to_numpy()
    ask_y, bid_y = df["yes_ask"].to_numpy(), df["yes_bid"].to_numpy()
    ask_n = 1 - bid_y
    ev_y = p - ask_y - feefn(ask_y)
    ev_n = (1 - p) - ask_n - feefn(ask_n)
    buy_y = (ev_y > edge) & (ev_y >= ev_n) & (ask_y < 1)
    buy_n = (ev_n > edge) & (ev_n > ev_y) & (ask_n < 1)
    y = df["outcome"].to_numpy()
    pnl = np.where(buy_y, y - ask_y - feefn(ask_y), np.where(buy_n, (1 - y) - ask_n - feefn(ask_n), np.nan))
    out = df.assign(side=np.where(buy_y, "yes", np.where(buy_n, "no", "")), pnl=pnl,
                    exp_edge=np.where(buy_y, ev_y, np.where(buy_n, ev_n, np.nan)))
    return out[out["side"] != ""]
