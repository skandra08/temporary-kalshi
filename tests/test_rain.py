import numpy as np
import pandas as pd
import pytest
from digitaledge import recurring as R, rain_study as S


def _entries(n_days=60, n_city=10, true_p=0.2, priced=0.3, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2026-01-01", periods=n_days, tz="UTC")
    rows = []
    for day in d:
        shared = rng.random()          # same-day weather shock: cities are correlated
        for c in range(n_city):
            y = int(rng.random() < true_p) if rng.random() < .5 else int(shared < true_p)
            rows.append(dict(date=day, city=f"c{c}", yes=y, yes_bid=priced - .005, yes_ask=priced + .005,
                             lead_h=24, ticker=f"{day:%m%d}-{c}", close_time=day))
    return pd.DataFrame(rows)


def test_buy_no_pnl_values():
    e = pd.DataFrame(dict(yes=[0, 1], yes_bid=[.2, .2], yes_ask=[.22, .22]))
    p = R.pnl_buy_no(e)
    ask_no = 0.8
    fee = 0.07 * ask_no * (1 - ask_no)
    assert p.iloc[0] == pytest.approx(1 - ask_no - fee)
    assert p.iloc[1] == pytest.approx(0 - ask_no - fee)


def test_overpriced_yes_makes_buy_no_profitable_and_fair_pricing_does_not():
    over = S.naive_no_table(_entries(true_p=0.2, priced=0.35, seed=1)).iloc[0]
    fair = S.naive_no_table(_entries(true_p=0.3, priced=0.3, seed=2)).iloc[0]
    assert over.pnl > 0.05 and over.lo > 0
    assert fair.pnl < 0.01 and fair.lo < 0.02      # fees make fair pricing slightly negative


def test_walk_forward_model_uses_no_future_outcomes():
    e = _entries(n_days=45, seed=3)
    rng = np.random.default_rng(4)
    feats = pd.DataFrame(dict(ticker=e["ticker"], fc_mm=rng.exponential(1, len(e)), fc_wet_hours=rng.integers(0, 10, len(e))))
    base = S.walk_forward_model(e, feats, min_train_days=20)
    day = np.sort(base["date"].unique())[10]
    e2 = e.copy()
    e2.loc[e2["date"] >= day, "yes"] = 1 - e2.loc[e2["date"] >= day, "yes"]      # flip the future
    shifted = S.walk_forward_model(e2, feats, min_train_days=20)
    early = base["date"] < day
    a = base[early].set_index("ticker")["p_model"]
    b = shifted[shifted["date"] < day].set_index("ticker")["p_model"]
    assert np.allclose(a.loc[b.index], b)
