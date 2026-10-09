import numpy as np
from digitaledge.conformal import Ladder, aci, central_score, randomized_pit, split_conformal


def _lad(cdf, lo):
    return Ladder("e", "s", 0.0, np.arange(len(cdf), dtype=float), np.asarray(cdf, float), lo, 1.0)


def test_pit_interval_and_quantile():
    l = _lad([0.2, 0.7], 1)
    assert l.pit_interval() == (0.2, 0.7)
    assert abs(l.quantile(0.45) - 0.5) < 1e-9


def test_calibrated_market_has_nominal_coverage_and_overdispersed_is_fixed():
    rng = np.random.default_rng(0)
    u = rng.random(20000)                      # perfectly calibrated PIT
    s = central_score(u)
    q = split_conformal(s[:5000], 0.1)
    assert abs((s[5000:] <= q).mean() - 0.9) < 0.01
    # market too narrow: realised PIT piles in the tails; nominal central 90% set undercovers badly
    u2 = np.where(rng.random(20000) < 0.5, rng.beta(0.5, 3, 20000) , 1 - rng.beta(0.5, 3, 20000))
    s2 = central_score(u2)
    nominal = (s2 <= 0.45).mean()
    q2 = split_conformal(s2[:5000], 0.1)
    assert nominal < 0.8 and abs((s2[5000:] <= q2).mean() - 0.9) < 0.015


def test_aci_recovers_under_shift():
    rng = np.random.default_rng(1)
    s = np.concatenate([np.abs(rng.random(3000) - 0.5), np.abs(rng.beta(0.4, 0.4, 3000) - 0.5)])
    miss, _ = aci(s, 0.1, gamma=0.02, warm=100)
    assert abs(miss.mean() - 0.1) < 0.02
    q = split_conformal(s[:100], 0.1)
    assert (s[3000:] > q).mean() > 0.15        # static split calibration breaks under shift
