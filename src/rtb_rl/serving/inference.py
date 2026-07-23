"""Bid scorer — the shared inference core for serving *and* offline evaluation.

Given a (website, user, placement) and a candidate ad set, it builds the state, assembles each
candidate's ad features (including cold-start ads scored via borrowed id-embeddings), runs a
single batched forward pass, and returns the argmax ad plus a suggested bid. Pure torch/numpy
(no FastAPI), so :mod:`rtb_rl.sim` reuses it to evaluate the learned policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rtb_rl.config import Config
from rtb_rl.features import encode
from rtb_rl.features.store import FeatureSnapshot
from rtb_rl.registry import ModelMeta
from rtb_rl.rl.cold_start import ColdStartResolver
from rtb_rl.rl.networks import QModel


@dataclass
class ColdAd:
    """A brand-new ad with no logged history, scored via cold-start borrowing."""

    ad_id: str
    content_emb: np.ndarray  # L2-normalized content embedding
    bid_cap_jpy: float


class UnknownCandidateError(ValueError):
    """A requested candidate ad id is not in the feature snapshot.

    Raised by :meth:`BidScorer.score` when ``candidate_ad_ids`` contains an id the snapshot
    has no row for. The serving layer maps this to an HTTP 400 (caller error) rather than
    letting the implicit ``KeyError`` surface as a 500.
    """


@dataclass
class ScoreResult:
    ad_id: str
    q_value: float
    bid_price_jpy: float
    cold_start: bool
    q_by_ad: dict[str, float]


def suggest_bid(q_value: float, bid_cap: float, floor: float) -> float:
    """Value-based bid: lean toward the cap as predicted click value rises, never below floor."""
    frac = float(np.clip(q_value, 0.0, 1.0))
    return float(np.clip(floor + bid_cap * frac, floor, bid_cap))


class BidScorer:
    def __init__(
        self, snap: FeatureSnapshot, model: QModel, meta: ModelMeta, cfg: Config,
        cold_k: int = 5,
    ) -> None:
        self.snap = snap
        self.model = model.eval()
        self.meta = meta
        self.cfg = cfg
        # Id-embedding rows are positional: row i is the learned vector for meta.ad_ids[i],
        # and the snapshot indexes ads by snap.ad_ids[i]. If the two orderings disagree (e.g. a
        # hot-swap paired a new model with a stale snapshot whose inventory changed), rows
        # silently misalign or index out of range. Refuse to score rather than emit garbage.
        if list(meta.ad_ids) != list(snap.ad_ids):
            raise ValueError(
                f"Model/snapshot ad-id mismatch: model {meta.version} was trained on "
                f"{len(meta.ad_ids)} ads, snapshot has {len(snap.ad_ids)}. "
                "Id-embedding rows are positional; reload features and the model together."
            )
        id_table = model.id_emb.weight.detach().cpu().numpy()
        self.resolver = ColdStartResolver(snap, id_table, k=cold_k)
        self._id_table = id_table  # (Na, id_dim)
        self._ad_index = {aid: i for i, aid in enumerate(snap.ad_ids)}

    @torch.inference_mode()
    def score(
        self,
        website_id: str,
        user_id: str,
        placement: str,
        candidate_ad_ids: list[str] | None = None,
        cold_candidates: list[ColdAd] | None = None,
        floor_price_jpy: float | None = None,
    ) -> ScoreResult:
        snap = self.snap
        user_vec, user_cold = self.resolver.user_vector(user_id, website_id)
        website_vec = snap.website_vec(website_id)
        affinity = float(user_vec @ website_vec)
        w_i = snap._w_idx[website_id]
        win, clear = snap.market_features(website_id, placement)
        state = encode.encode_state_np(
            snap, website_vec, user_vec, affinity,
            float(snap.website_base_ctr[w_i]), float(snap.website_base_cvr[w_i]),
            placement, win, clear,
        )

        ad_ids, ad_emb, smoothed, bidcap, is_cold, id_emb = self._gather_candidates(
            candidate_ad_ids, cold_candidates
        )
        ad_content = encode.assemble_ad_content_np(
            ad_emb, website_vec, user_vec, smoothed, bidcap,
            budget_ratio=np.ones(len(ad_ids), dtype=np.float32), is_cold=is_cold,
        )

        state_t = torch.from_numpy(state).unsqueeze(0)  # (1, Ds)
        ac_t = torch.from_numpy(ad_content).unsqueeze(0)  # (1, M, Dac)
        id_t = torch.from_numpy(id_emb).unsqueeze(0)  # (1, M, id_dim)
        q = self.model.q_values_emb(state_t, ac_t, id_t)[0].cpu().numpy()  # (M,)

        best = int(np.argmax(q))
        floor = self.cfg.serving.default_floor_jpy if floor_price_jpy is None else floor_price_jpy
        return ScoreResult(
            ad_id=ad_ids[best],
            q_value=float(q[best]),
            bid_price_jpy=suggest_bid(float(q[best]), float(bidcap[best]), floor),
            cold_start=bool(user_cold or is_cold[best]),
            q_by_ad={aid: float(qi) for aid, qi in zip(ad_ids, q, strict=True)},
        )

    def _gather_candidates(
        self, candidate_ad_ids: list[str] | None, cold_candidates: list[ColdAd] | None
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        snap = self.snap
        if candidate_ad_ids is None:
            rows = np.arange(len(snap.ad_ids))
        else:
            unknown = [a for a in candidate_ad_ids if a not in self._ad_index]
            if unknown:
                raise UnknownCandidateError(
                    f"Unknown candidate ad ids (not in snapshot): {unknown}"
                )
            rows = np.array([self._ad_index[a] for a in candidate_ad_ids], dtype=int)

        ad_ids = [snap.ad_ids[i] for i in rows]
        ad_emb = snap.ad_emb[rows]
        smoothed = snap.ad_smoothed_ctr[rows]
        bidcap = snap.ad_bid_cap[rows]
        is_cold = np.zeros(len(rows), dtype=np.float32)
        id_emb = self._id_table[rows]

        if cold_candidates:
            c_emb = np.stack([c.content_emb for c in cold_candidates]).astype(np.float32)
            c_id = np.stack(
                [self.resolver.id_embedding_for(c.content_emb) for c in cold_candidates]
            )
            c_ctr = np.array(
                [self.resolver.smoothed_ctr_for(c.content_emb) for c in cold_candidates],
                dtype=np.float32,
            )
            ad_ids = ad_ids + [c.ad_id for c in cold_candidates]
            ad_emb = np.concatenate([ad_emb, c_emb], axis=0)
            smoothed = np.concatenate([smoothed, c_ctr], axis=0)
            bidcap = np.concatenate([bidcap, np.array([c.bid_cap_jpy for c in cold_candidates],
                                                      dtype=np.float32)], axis=0)
            is_cold = np.concatenate([is_cold, np.ones(len(cold_candidates), dtype=np.float32)])
            id_emb = np.concatenate([id_emb, c_id.astype(np.float32)], axis=0)

        return ad_ids, ad_emb, smoothed, bidcap, is_cold, id_emb
