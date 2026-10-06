import numpy as np
import pandas as pd
import pytest
from digitaledge.temperature import (bracket_bounds, bracket_prob, observed_temp,
                                     walk_forward_error_model)


def test_brackets_partition_the_line():
    mk = [("less", np.nan, 64.0), ("between", 64.0, 65.0), ("between", 66.0, 67.0),
          ("between", 68.0, 69.0), ("between", 70.0, 71.0), ("greater", 71.0, np.nan)]
    # 'less 64' = 63 or below -> (-inf, 63.5); ladder above should tile with the 64-65 bracket etc.
    p = [bracket_prob(67.0, 3.0, *bracket_bounds(*m)) for m in mk]
    assert sum(p) == pytest.approx(1.0, abs=0.12)       # 62..., 65.5-66.5 and 67.5-... gaps (odd-only ladder)
    full = [("less", np.nan, 64.0), ("between", 64.0, 65.0), ("between", 66.0, 67.0),
            ("between", 68.0, 69.0), ("between", 70.0, 71.0), ("between", 72.0, 73.0), ("greater", 73.0, np.nan)]
    lo_hi = [bracket_bounds(*m) for m in full]
    assert lo_hi[0][1] == 63.5 and lo_hi[1] == (63.5, 65.5) and lo_hi[-1][0] == 73.5


def test_integer_rounding_boundaries():
    assert bracket_bounds("between", 70.0, 71.0) == (69.5, 71.5)
    assert bracket_bounds("greater", 71.0, np.nan) == (71.5, np.inf)      # '72 or above'
    assert bracket_bounds("less", np.nan, 64.0) == (-np.inf, 63.5)        # '63 or below'


def test_prob_symmetry_and_total():
    assert bracket_prob(70, 2, 69.5, 70.5) > bracket_prob(70, 2, 71.5, 72.5)
    assert bracket_prob(70, 2, -np.inf, np.inf) == pytest.approx(1.0)
    assert bracket_prob(70, 2, -np.inf, 70) == pytest.approx(0.5)


def test_observed_temp_midpoint_and_censoring():
    ev = pd.DataFrame(dict(result=["no", "yes", "no"], strike_type=["greater", "between", "less"],
                           floor_strike=[71, 68, np.nan], cap_strike=[np.nan, 69, 64]))
    assert observed_temp(ev) == 68.5
    ev.loc[0, ["result"]] = "yes"; ev.loc[1, ["result"]] = "no"
    assert np.isnan(observed_temp(ev))


def test_error_model_recovers_bias_and_sigma_without_peeking():
    rng = np.random.default_rng(0)
    n = 200
    fc = 60 + 10 * rng.random(n)
    obs = fc + 1.5 + 2.0 * rng.normal(size=n)
    df = pd.DataFrame(dict(city="X", date=pd.date_range("2026-01-01", periods=n, tz="UTC"), fc=fc, obs=obs))
    out = walk_forward_error_model(df).iloc[100:]
    assert (out["mu"] - out["fc"]).mean() == pytest.approx(1.5, abs=0.5)
    assert out["sigma"].mean() == pytest.approx(2.0, abs=0.4)
    # perturb the future: earlier rows must not change
    df2 = df.copy(); df2.loc[df2.index >= 150, "obs"] += 10
    a = walk_forward_error_model(df).iloc[:150][["mu", "sigma"]].to_numpy()
    b = walk_forward_error_model(df2).iloc[:150][["mu", "sigma"]].to_numpy()
    assert np.allclose(a, b)
