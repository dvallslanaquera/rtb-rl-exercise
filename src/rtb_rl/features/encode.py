"""Shared state/action feature encoding.

Training, simulation and serving MUST encode features identically, so the formulas live here
once. The trainer mirrors :func:`assemble_ad_content_np` in batched torch ops (see
``rtb_rl/rl/replay.py`` / ``trainer.py``) — keep the two in sync.

State (per bid request), dim = 2E + 5 + P:
    [ website_emb(E) | user_emb(E) | affinity, base_ctr, base_cvr, mkt_win_rate, mkt_clear_norm
      | placement_onehot(P) ]

Ad action features (per (state, ad)), dim = E + 6:
    [ ad_emb(E) | site_match, user_match, smoothed_ctr, bid_cap_norm, budget_ratio, is_cold ]

``site_match``/``user_match`` are cosine similarities (embeddings are L2-normalized) — exactly
the recoverable signals the latent click model is built from. A learned per-ad id-embedding is
concatenated to the ad features inside the Q-model (handles the residual ad appeal + cold start).
"""

from __future__ import annotations

import math

import numpy as np

from rtb_rl.features.store import FeatureSnapshot

BID_CAP_NORM = 300.0  # JPY; normalizes bid caps to ~[0, 1]
CLEAR_NORM = math.log1p(2000.0)  # normalizes log clearing price
N_STATE_EXTRA = 5  # affinity, base_ctr, base_cvr, win_rate, clear_norm
N_AD_EXTRA = 6  # site_match, user_match, smoothed_ctr, bid_cap_norm, budget_ratio, is_cold


def state_dim(snap: FeatureSnapshot) -> int:
    return 2 * snap.embed_dim + N_STATE_EXTRA + len(snap.placements)


def ad_content_dim(snap: FeatureSnapshot) -> int:
    return snap.embed_dim + N_AD_EXTRA


def placement_onehot(placements: list[str], placement: str) -> np.ndarray:
    vec = np.zeros(len(placements), dtype=np.float32)
    if placement in placements:
        vec[placements.index(placement)] = 1.0
    return vec


def clear_norm(avg_clear_jpy: float) -> float:
    return float(math.log1p(max(avg_clear_jpy, 0.0)) / CLEAR_NORM)


def encode_state_np(
    snap: FeatureSnapshot,
    website_vec: np.ndarray,
    user_vec: np.ndarray,
    affinity: float,
    base_ctr: float,
    base_cvr: float,
    placement: str,
    market_win: float,
    market_clear: float,
) -> np.ndarray:
    extra = np.array(
        [affinity, base_ctr, base_cvr, market_win, clear_norm(market_clear)], dtype=np.float32
    )
    return np.concatenate(
        [website_vec, user_vec, extra, placement_onehot(snap.placements, placement)]
    ).astype(np.float32)


def assemble_ad_content_np(
    ad_emb: np.ndarray,  # (M, E)
    website_vec: np.ndarray,  # (E,)
    user_vec: np.ndarray,  # (E,)
    smoothed_ctr: np.ndarray,  # (M,)
    bid_cap: np.ndarray,  # (M,) raw JPY
    budget_ratio: np.ndarray,  # (M,) in [0, 1]
    is_cold: np.ndarray,  # (M,) bool/float
) -> np.ndarray:
    """Per-candidate ad content features, dim (M, E + 6)."""
    site_match = ad_emb @ website_vec  # (M,)
    user_match = ad_emb @ user_vec  # (M,)
    scalars = np.stack(
        [
            site_match,
            user_match,
            smoothed_ctr.astype(np.float32),
            (bid_cap / BID_CAP_NORM).astype(np.float32),
            budget_ratio.astype(np.float32),
            is_cold.astype(np.float32),
        ],
        axis=1,
    )
    return np.concatenate([ad_emb.astype(np.float32), scalars], axis=1).astype(np.float32)
