"""Offline dataset for batch RL, built from the feature snapshot + logged auctions.

Only **won** impressions are used: a click is observable only when the ad was actually shown,
so ``Q(s, a)`` learns expected click value *conditional on winning* — exactly the
click-probability ranking the bidder needs. Each transition is one-step / terminal
(independent impressions); the Double-DQN bootstrap path exists for the simulator's
sequential, budget-paced episodes (``gamma > 0``) used during retraining.

State encoding is built here in vectorized numpy, mirroring
:func:`rtb_rl.features.encode.encode_state_np`. Per-minibatch candidate pools place the logged
action in column 0 so its Q is trivially indexable for the TD target and the CQL penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

from rtb_rl.config import Config
from rtb_rl.data import loaders
from rtb_rl.features import encode
from rtb_rl.features.store import FeatureSnapshot


@dataclass
class OfflineDataset:
    states: np.ndarray  # (N, Ds)
    w_emb: np.ndarray  # (N, E)
    u_emb: np.ndarray  # (N, E)
    action_idx: np.ndarray  # (N,) logged ad row index
    reward: np.ndarray  # (N,)
    # snapshot ad arrays (known ads), shared by all transitions
    ad_emb: np.ndarray  # (Na, E)
    ad_smoothed: np.ndarray  # (Na,)
    ad_bid_cap: np.ndarray  # (Na,)
    state_dim: int = 0
    ad_content_dim: int = 0
    n_ads: int = 0

    def __post_init__(self) -> None:
        self.state_dim = self.states.shape[1]
        self.n_ads = self.ad_emb.shape[0]
        self.ad_content_dim = self.ad_emb.shape[1] + encode.N_AD_EXTRA
        self._ad_emb_t = torch.from_numpy(self.ad_emb)
        self._smoothed_t = torch.from_numpy(self.ad_smoothed)
        self._bidcap_t = torch.from_numpy(self.ad_bid_cap)

    def __len__(self) -> int:
        return self.states.shape[0]

    def sample_batch(
        self, batch_size: int, candidate_pool: int, rng: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        n = len(self)
        rows = rng.integers(0, n, size=batch_size)
        c = max(2, candidate_pool)
        # column 0 = logged action; remaining columns = random known ads
        rand = rng.integers(0, self.n_ads, size=(batch_size, c - 1))
        pool_idx = np.concatenate([self.action_idx[rows, None], rand], axis=1)  # (B, C)
        pool_t = torch.from_numpy(pool_idx).long()

        states = torch.from_numpy(self.states[rows])  # (B, Ds)
        w = torch.from_numpy(self.w_emb[rows])  # (B, E)
        u = torch.from_numpy(self.u_emb[rows])  # (B, E)
        ad_emb = self._ad_emb_t[pool_t]  # (B, C, E)

        site_match = torch.einsum("bce,be->bc", ad_emb, w)
        user_match = torch.einsum("bce,be->bc", ad_emb, u)
        smoothed = self._smoothed_t[pool_t]
        bidcap = self._bidcap_t[pool_t] / encode.BID_CAP_NORM
        budget = torch.ones_like(smoothed)
        is_cold = torch.zeros_like(smoothed)
        scalars = torch.stack([site_match, user_match, smoothed, bidcap, budget, is_cold], dim=-1)
        ad_content = torch.cat([ad_emb, scalars], dim=-1)  # (B, C, E+6)

        return {
            "state": states,
            "ad_content": ad_content,
            "pool_idx": pool_t,
            "reward": torch.from_numpy(self.reward[rows].astype(np.float32)),
        }


def build_offline_dataset(cfg: Config, snap: FeatureSnapshot) -> OfflineDataset:
    logs = loaders.load_bid_logs_df(cfg).filter(pl.col("won"))
    a_idx = {aid: i for i, aid in enumerate(snap.ad_ids)}
    w_idx = {wid: i for i, wid in enumerate(snap.website_ids)}
    u_idx = {uid: i for i, uid in enumerate(snap.user_ids)}

    # keep rows whose logged ad is a known ad (cold-start ads never appear in logs anyway)
    logs = logs.filter(pl.col("ad_id").is_in(list(a_idx.keys())))
    wid = logs["website_id"].to_list()
    uid = logs["user_id"].to_list()
    placement = logs["placement"].to_list()
    ad_id = logs["ad_id"].to_list()
    click = logs["click"].to_numpy().astype(np.float32)
    cost = logs["cost_jpy"].to_numpy().astype(np.float32)

    w_rows = np.array([w_idx[x] for x in wid])
    u_rows = np.array([u_idx[x] for x in uid])
    action_idx = np.array([a_idx[x] for x in ad_id])

    w_emb = snap.website_emb[w_rows]
    u_emb = snap.user_emb[u_rows]
    affinity = np.einsum("ne,ne->n", u_emb, w_emb).astype(np.float32)
    base_ctr = snap.website_base_ctr[w_rows]
    base_cvr = snap.website_base_cvr[w_rows]

    p_index = {p: i for i, p in enumerate(snap.placements)}
    onehot = np.zeros((len(placement), len(snap.placements)), dtype=np.float32)
    for i, p in enumerate(placement):
        onehot[i, p_index[p]] = 1.0

    win = np.empty(len(placement), dtype=np.float32)
    clear = np.empty(len(placement), dtype=np.float32)
    for i in range(len(placement)):
        w_rate, c_avg = snap.market_features(wid[i], placement[i])
        win[i] = w_rate
        clear[i] = encode.clear_norm(c_avg)

    extra = np.stack([affinity, base_ctr, base_cvr, win, clear], axis=1).astype(np.float32)
    states = np.concatenate([w_emb, u_emb, extra, onehot], axis=1).astype(np.float32)

    reward = (cfg.rl.click_reward * click - cfg.rl.cost_coef * cost).astype(np.float32)

    return OfflineDataset(
        states=states,
        w_emb=w_emb.astype(np.float32),
        u_emb=u_emb.astype(np.float32),
        action_idx=action_idx,
        reward=reward,
        ad_emb=snap.ad_emb.astype(np.float32),
        ad_smoothed=snap.ad_smoothed_ctr.astype(np.float32),
        ad_bid_cap=snap.ad_bid_cap.astype(np.float32),
    )
