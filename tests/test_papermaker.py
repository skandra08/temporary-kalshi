import pandas as pd
from digitaledge.papermaker import simulate_market


def _books(rows):
    return pd.DataFrame([dict(ts=pd.Timestamp(t, tz="UTC"), ticker="M", yes_ask=a, no=no) for t, a, no in rows])


def _trades(rows):
    return pd.DataFrame([dict(ts=pd.Timestamp(t, tz="UTC"), yes_price=p, count=c, taker_side=s) for t, p, c, s in rows])


def test_fills_only_after_queue_ahead_is_consumed():
    b = _books([("2026-01-01 00:00:00", 0.10, [[0.90, 100.0]]), ("2026-01-01 00:10:00", 0.10, [[0.90, 40.0]])])
    t = _trades([("2026-01-01 00:01:00", 0.10, 60.0, "yes"), ("2026-01-01 00:02:00", 0.10, 50.0, "yes")])
    g = simulate_market(b, t)[0]
    assert g.depth_ahead == 100.0 and g.filled                 # 60 then +50 = 110 >= 100 + 1
    t2 = _trades([("2026-01-01 00:01:00", 0.10, 60.0, "yes"), ("2026-01-01 00:02:00", 0.10, 30.0, "yes")])
    assert not simulate_market(b, t2)[0].filled                # 90 < 101: still queued


def test_wrong_side_and_wrong_price_do_not_fill():
    b = _books([("2026-01-01 00:00:00", 0.10, [[0.90, 10.0]]), ("2026-01-01 00:30:00", 0.10, [[0.90, 10.0]])])
    t = _trades([("2026-01-01 00:01:00", 0.10, 500.0, "no"), ("2026-01-01 00:02:00", 0.11, 500.0, "yes")])
    assert not simulate_market(b, t)[0].filled


def test_cancelled_when_ask_rises_so_later_trades_do_not_fill():
    b = _books([("2026-01-01 00:00:00", 0.10, [[0.90, 5.0]]), ("2026-01-01 00:05:00", 0.12, [[0.88, 5.0]])])
    t = _trades([("2026-01-01 00:06:00", 0.10, 500.0, "yes")])        # after the cancel
    assert not simulate_market(b, t)[0].filled


def test_prices_outside_band_are_not_quoted():
    b = _books([("2026-01-01 00:00:00", 0.02, [[0.98, 5.0]]), ("2026-01-01 00:01:00", 0.60, [[0.40, 5.0]])])
    assert simulate_market(b, _trades([])) == []


def test_no_fill_after_sixty_minutes():
    b = _books([("2026-01-01 00:00:00", 0.10, [[0.90, 5.0]]), ("2026-01-01 02:00:00", 0.10, [[0.90, 5.0]])])
    t = _trades([("2026-01-01 01:30:00", 0.10, 500.0, "yes")])
    assert not simulate_market(b, t)[0].filled
