"""Paper-trading a passive NO bidder (PREREGISTRATION.md addendum H2).

Ghost order: at a book snapshot whose best YES ask is `a` (0.03 <= a <= 0.40), rest a bid for 1 NO at 1-a, behind the
displayed NO depth D at that level. It fills when cumulative taker-YES-buy volume at yes-price a after posting exceeds D;
it cancels if the best ask rises above a, after 60 minutes, or at the market's last snapshot. P&L per fill = a - y."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

LO, HI, MAX_MIN = 0.03, 0.40, 60


@dataclass
class Ghost:
    ticker: str
    t_post: pd.Timestamp
    ask: float
    depth_ahead: float
    filled: bool
    t_fill: pd.Timestamp = None
    fill_trade_size: float = np.nan


def _level_qty(levels, price):
    for p, q in levels:
        if abs(p - price) < 1e-9:
            return q
    return 0.0


def simulate_market(books, trades, lo=LO, hi=HI, max_min=MAX_MIN):
    """books: DataFrame[ts, yes_ask, no(list of [p,q])] sorted by ts; trades: DataFrame[ts, yes_price, count, taker_side] sorted by ts."""
    out = []
    bt = books.reset_index(drop=True)
    if trades is None or len(trades) == 0:
        trades = pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns, UTC]"), "yes_price": pd.Series(dtype=float),
                               "count": pd.Series(dtype=float), "taker_side": pd.Series(dtype=object)})
    tr = trades[trades["taker_side"] == "yes"].reset_index(drop=True)
    for i, b in bt.iterrows():
        a = b["yes_ask"]
        if a is None or not (lo <= a <= hi):
            continue
        depth = _level_qty(b["no"], round(1 - a, 4))
        t0, deadline = b["ts"], b["ts"] + pd.Timedelta(minutes=max_min)
        cancel = deadline
        for j in range(i + 1, len(bt)):                       # first later snapshot where the ask rose above a (level gone)
            nb = bt.iloc[j]
            if nb["ts"] > deadline:
                break
            if nb["yes_ask"] is not None and nb["yes_ask"] > a + 1e-9:
                cancel = nb["ts"]
                break
        else:
            cancel = min(deadline, bt["ts"].iloc[-1])
        seg = tr[(tr["ts"] > t0) & (tr["ts"] <= cancel) & (np.abs(tr["yes_price"] - a) < 1e-9)]
        cum, g = 0.0, Ghost(b["ticker"], t0, a, depth, False)
        for r in seg.itertuples():
            cum += r.count
            if cum >= depth + 1.0:
                g.filled, g.t_fill, g.fill_trade_size = True, r.ts, r.count
                break
        out.append(g)
    return out
