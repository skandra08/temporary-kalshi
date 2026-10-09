import numpy as np
from crowdskill.skill import Confusion, fit_logit, predict, replay_features


def _rec(i, t, votes, y, p=0.5):
    return dict(id=str(i), anchor_t=t, resolve_t=t + 5, y=y, p_anchor=p, votes=votes)


def test_llr_signs_and_unseen_zero():
    c = Confusion(prior=1.0)
    for _ in range(40): c.add({"good": 1, "bad": 1}, 1)
    for _ in range(40): c.add({"good": -1, "bad": 1}, 0)
    assert c.llr("good", 1) > 1 and c.llr("good", -1) < -1
    assert abs(c.llr("bad", 1)) < 0.1 and c.llr("nobody", 1) == 0.0


def test_replay_has_no_lookahead():
    # market 0 resolves at t=105; market 1 anchors at t=100 (must NOT see it), market 2 at t=200 (must)
    recs = [_rec(0, 100, {"a": 1}, 1), _rec(1, 100.5, {"a": 1}, 1), _rec(2, 200, {"a": 1}, 1)]
    recs[0]["resolve_t"] = 105
    f = replay_features(recs, prior=1.0)
    assert f.evidence[1] == 0.0 and f.n_known[1] == 0
    assert f.n_known[2] == 1


def test_logit_recovers_weights():
    rng = np.random.default_rng(0); X = rng.normal(size=(20000, 2)); w = np.array([0.3, 1.0, -0.5])
    y = (rng.random(20000) < 1 / (1 + np.exp(-(w[0] + X @ w[1:])))).astype(float)
    assert np.abs(fit_logit(X, y, l2=1e-6) - w).max() < 0.08
    assert 0 < predict(fit_logit(X, y), X).min()
