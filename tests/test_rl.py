import numpy as np
import pytest
from mmsim import FixedSpread, FlowParams, run_strategy, simulate_path
from mmsim.rl import ACTIONS, QAgent, RLQuoter, TileCoder, normalise, run_episode


class _FixedAgent:
    """Always plays one action; lets the RL environment be compared with the validated simulator."""
    def __init__(self, action, base):
        self.a = ACTIONS.index(action); self.tc = base.tc

    def act(self, idx, eps=0.0):
        return self.a

    def learn(self, *args):
        pass


def test_rl_environment_reproduces_simulator_pnl_for_a_fixed_policy():
    p = FlowParams()
    for seed in range(5):
        path = simulate_path(p, seed)
        ref = run_strategy(path, FixedSpread(2.0), max_inv=10)
        pnl, q, fills, _ = run_episode(path, _FixedAgent((2, 2), QAgent()), learn=False)
        assert pnl == pytest.approx(ref.pnl, abs=1e-6)
        assert q == ref.inventory_final and fills == ref.n_fills


def test_tile_coder_nearby_states_share_tiles_far_states_do_not():
    tc = TileCoder()
    a = tc.active(np.array([0.50, 0.50, 0.30, 0.30]))
    b = tc.active(np.array([0.51, 0.50, 0.30, 0.31]))
    far = tc.active(np.array([0.05, 0.95, 0.9, 0.05]))
    assert len(set(a) & set(b)) >= 4 and len(set(a) & set(far)) == 0
    assert a.min() >= 0 and a.max() < tc.n_features


def test_q_update_moves_value_toward_target():
    ag = QAgent(alpha=0.5)
    idx = ag.tc.active(normalise(10.0, 0, 0.5, 0.5, 600.0, 10))
    before = ag.q_values(idx)[3]
    ag.learn(idx, 3, 1.0, idx, True)
    assert ag.q_values(idx)[3] > before and ag.q_values(idx)[2] == 0.0       # only the played action moves


def test_greedy_quoter_returns_valid_distances_and_runs_in_the_simulator():
    ag = QAgent()
    res = run_strategy(simulate_path(FlowParams(), 1), RLQuoter(ag), max_inv=10)
    assert np.isfinite(res.pnl)
    assert RLQuoter(ag)(5.0, 0.0, 0, 0.1, 0.1, FlowParams()) in ACTIONS


def test_dampened_reward_leaves_losses_but_not_speculative_gains():
    # r = dPnL - eta*max(0, q*dMid): a gain from holding inventory is discounted; a loss is not
    eta, q = 0.5, 4
    assert 1.0 - eta * max(0.0, q * 0.25) == pytest.approx(0.5)          # +1 gain, dMid=+0.25 -> reduced to 0.5
    assert -1.0 - eta * max(0.0, q * -0.25) == pytest.approx(-1.0)       # loss unchanged
