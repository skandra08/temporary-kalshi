"""Reinforcement-learning market maker on the Hawkes-flow simulator (after Spooner et al., 2018, "Market Making via RL").

State  : (inventory, time fraction, buy excitation, sell excitation), tile-coded.
Action : a pair of quote distances (bid, ask) in ticks from a small grid.
Reward : change in mark-to-market wealth, asymmetrically dampened: r = dPnL - eta * max(0, q * dMid), which keeps losses from
         inventory but discounts speculative inventory gains (the paper's "ADP" reward). Evaluation always uses true P&L.
Learner: linear Q-learning with epsilon-greedy exploration (numpy only).
"""
from dataclasses import dataclass
import numpy as np
from .flow import FlowParams, Path, simulate_path

DISTS = (1, 2, 3, 5)
ACTIONS = [(b, a) for b in DISTS for a in DISTS]            # (bid distance, ask distance) in ticks


class TileCoder:
    def __init__(self, bins=(6, 4, 4, 4), n_tilings=8, seed=0):
        self.bins = np.array(bins)
        self.n_tilings = n_tilings
        rng = np.random.default_rng(seed)
        self.offsets = rng.random((n_tilings, len(bins)))      # in units of one tile
        self.grid = self.bins + 1
        self.tiles_per_tiling = int(np.prod(self.grid))
        self.n_features = n_tilings * self.tiles_per_tiling
        self._mult = np.concatenate([[1], np.cumprod(self.grid[:-1])])

    def active(self, u):
        """u: 4 normalised coordinates in [0, 1] -> indices of the active tile in each tiling."""
        coords = np.floor(np.asarray(u) * self.bins + self.offsets).astype(int)      # (tilings, dims)
        coords = np.minimum(coords, self.grid - 1)
        return coords @ self._mult + np.arange(self.n_tilings) * self.tiles_per_tiling


def normalise(t, q, eb, es, horizon, max_inv):
    sq = lambda x: x / (x + 1.0)
    return np.array([(q + max_inv) / (2.0 * max_inv), min(t / horizon, 1.0), sq(eb), sq(es)])


class QAgent:
    def __init__(self, max_inv=10, alpha=0.1, gamma=0.999, seed=0, tiles=None):
        self.max_inv, self.alpha, self.gamma = max_inv, alpha, gamma
        self.tc = tiles or TileCoder(seed=seed)
        self.W = np.zeros((len(ACTIONS), self.tc.n_features))
        self.rng = np.random.default_rng(seed)

    def q_values(self, idx):
        return self.W[:, idx].sum(axis=1)

    def act(self, idx, eps=0.0):
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.integers(len(ACTIONS)))
        return int(np.argmax(self.q_values(idx)))

    def learn(self, idx, a, r, idx_next, terminal):
        target = r if terminal else r + self.gamma * self.q_values(idx_next).max()
        delta = target - self.W[a, idx].sum()
        self.W[a, idx] += self.alpha / self.tc.n_tilings * delta
        return delta


def run_episode(path: Path, agent, eps=0.0, learn=True, max_inv=10, eta=0.5, phi=0.0):
    """Play one path. Returns (true terminal P&L, final inventory, number of fills, inventory std)."""
    p, n = path.params, len(path.t)
    cash, q, prev_value, fills = 0.0, 0, 0.0, 0
    inv = []
    if n == 0:
        return 0.0, 0, 0, 0.0
    idx = agent.tc.active(normalise(path.t[0], q, path.exc_buy[0], path.exc_sell[0], p.horizon, max_inv))
    for i in range(n):
        a = agent.act(idx, eps)
        bid_d, ask_d = ACTIONS[a]
        mid = path.mid_pre[i]
        if path.sign[i] == 1 and q > -max_inv and path.depth[i] > ask_d:
            cash += mid + ask_d; q -= 1; fills += 1
        elif path.sign[i] == -1 and q < max_inv and path.depth[i] > bid_d:
            cash -= mid - bid_d; q += 1; fills += 1
        inv.append(q)
        last = i == n - 1
        m_next = path.mid_post[-1] if last else path.mid_pre[i + 1]
        value = cash + q * m_next
        r = (value - prev_value) - eta * max(0.0, q * (m_next - mid)) - phi * q * q      # phi: per-step quadratic inventory penalty
        prev_value = value
        if not last:
            idx_next = agent.tc.active(normalise(path.t[i + 1], q, path.exc_buy[i + 1], path.exc_sell[i + 1], p.horizon, max_inv))
        else:
            idx_next = idx
        if learn:
            agent.learn(idx, a, r, idx_next, last)
        idx = idx_next
    return prev_value, q, fills, float(np.std(inv))


@dataclass
class RLQuoter:
    """Greedy policy of a trained agent, usable anywhere a strategy is (see mmsim.engine.run_strategy)."""
    agent: QAgent
    name: str = "rl-q"

    def __call__(self, t, mid, q, exc_buy, exc_sell, p):
        idx = self.agent.tc.active(normalise(t, q, exc_buy, exc_sell, p.horizon, self.agent.max_inv))
        return ACTIONS[self.agent.act(idx, 0.0)]
