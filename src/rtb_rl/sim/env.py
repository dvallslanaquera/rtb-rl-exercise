"""Gymnasium auction environment — the sequential, budget-paced MDP.

Each step is one bid opportunity (sampled website/user/placement) with a candidate pool of
ads that still have budget. The action selects an ad; the env simulates the auction (clearing
price ~ lognormal), the click (Bernoulli under the shared :class:`LatentClickModel`), and
**decrements the campaign budget** by the price paid. Budget depletion is what couples
successive impressions into an episode, making ``gamma > 0`` meaningful and justifying full
DQN over a one-step bandit. Used for policy roll-outs and optional online fine-tuning during
retraining; the headline offline metric lives in :mod:`rtb_rl.sim.evaluate`.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rtb_rl.config import Config
from rtb_rl.data import loaders
from rtb_rl.data.synth import LatentClickModel
from rtb_rl.features import encode
from rtb_rl.features.store import FeatureSnapshot

_PLACEMENT_PRICE = {
    "header": 120.0, "in_article": 90.0, "interstitial": 150.0, "sidebar": 60.0, "footer": 40.0,
}


class RTBSimEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, cfg: Config, snap: FeatureSnapshot, latent: LatentClickModel,
        horizon: int = 256, pool: int | None = None, seed: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.snap = snap
        self.latent = latent
        self.horizon = horizon
        self.pool = min(pool or cfg.sim.candidate_pool, len(snap.ad_ids))
        self.rng = np.random.default_rng(seed if seed is not None else cfg.seed)

        budgets = {a.ad_id: a.daily_budget_jpy for a in loaders.load_ads(cfg)}
        self._init_budget = np.array([budgets.get(a, 1e9) for a in snap.ad_ids], dtype=np.float64)

        ds = encode.state_dim(snap)
        dac = encode.ad_content_dim(snap)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-np.inf, np.inf, (ds,), np.float32),
                "ad_content": spaces.Box(-np.inf, np.inf, (self.pool, dac), np.float32),
                "ad_idx": spaces.Box(0, len(snap.ad_ids), (self.pool,), np.int64),
            }
        )
        self.action_space = spaces.Discrete(self.pool)
        self._budget = self._init_budget.copy()
        self._t = 0
        self._ctx: dict[str, Any] = {}

    # ---- gym API ----
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._budget = self._init_budget.copy()
        self._t = 0
        return self._new_observation(), {}

    def step(self, action: int):
        ctx = self._ctx
        row = ctx["rows"][int(action)]
        ad_id = self.snap.ad_ids[row]
        placement = ctx["placement"]

        market = float(np.clip(
            self.rng.lognormal(math.log(_PLACEMENT_PRICE.get(placement, 80.0)), 0.4), 5, 2000))
        bid = float(self.snap.ad_bid_cap[row])
        won = bid >= market and self._budget[row] >= market
        reward = 0.0
        if won:
            p = self.latent.p_click(ctx["user_id"], ad_id, ctx["website_id"], placement)
            click = float(self.rng.random() < p)
            self._budget[row] -= market
            reward = self.cfg.rl.click_reward * click - self.cfg.rl.cost_coef * market

        self._t += 1
        terminated = False
        truncated = self._t >= self.horizon
        obs = self._new_observation() if not truncated else self._ctx["obs"]
        info = {"won": won, "ad_id": ad_id, "market": market}
        return obs, reward, terminated, truncated, info

    # ---- helpers ----
    def _new_observation(self) -> dict[str, np.ndarray]:
        snap = self.snap
        wid = snap.website_ids[int(self.rng.integers(0, len(snap.website_ids)))]
        uid = snap.user_ids[int(self.rng.integers(0, len(snap.user_ids)))]
        placement = snap.placements[int(self.rng.integers(0, len(snap.placements)))]

        eligible = np.where(self._budget > snap.ad_bid_cap.astype(np.float64).min())[0]
        if len(eligible) < self.pool:
            eligible = np.arange(len(snap.ad_ids))
        rows = self.rng.choice(eligible, size=self.pool, replace=False)

        website_vec = snap.website_vec(wid)
        user_vec = snap.user_vec(uid)
        affinity = float(user_vec @ website_vec)
        w_i = snap._w_idx[wid]
        win, clear = snap.market_features(wid, placement)
        state = encode.encode_state_np(
            snap, website_vec, user_vec, affinity,
            float(snap.website_base_ctr[w_i]), float(snap.website_base_cvr[w_i]),
            placement, win, clear,
        )
        budget_ratio = (self._budget[rows] / np.maximum(self._init_budget[rows], 1.0)).astype(
            np.float32)
        ad_content = encode.assemble_ad_content_np(
            snap.ad_emb[rows], website_vec, user_vec,
            snap.ad_smoothed_ctr[rows], snap.ad_bid_cap[rows],
            budget_ratio=np.clip(budget_ratio, 0.0, 1.0),
            is_cold=np.zeros(len(rows), np.float32),
        )
        obs: dict[str, np.ndarray] = {
            "state": state.astype(np.float32),
            "ad_content": ad_content.astype(np.float32),
            "ad_idx": rows.astype(np.int64),
        }
        self._ctx = {"rows": rows, "website_id": wid, "user_id": uid,
                     "placement": placement, "obs": obs}
        return obs
