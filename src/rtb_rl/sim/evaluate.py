"""Offline evaluation of the learned bidding policy.

Headline metric: expected CTR of the learned (argmax-Q) policy vs. the logging/behavior policy
(random selection), with an oracle ceiling (best ad in the candidate pool under the true latent
click model). Using *expected* click probability — rather than sampled clicks — gives a
low-variance, reproducible uplift number. A SNIPS off-policy estimate from the logged data is
provided as a secondary, model-free check.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from rtb_rl.config import Config
from rtb_rl.data import loaders
from rtb_rl.data.synth import LatentClickModel
from rtb_rl.embeddings import get_embedder
from rtb_rl.features.store import FeatureSnapshot
from rtb_rl.serving.inference import BidScorer, ColdAd

logger = logging.getLogger(__name__)


def make_eval_requests(
    cfg: Config, snap: FeatureSnapshot, rng: np.random.Generator, n: int, pool: int
) -> list[dict]:
    pool = min(pool, len(snap.ad_ids))
    reqs: list[dict] = []
    for i in range(n):
        wid = snap.website_ids[int(rng.integers(0, len(snap.website_ids)))]
        uid = snap.user_ids[int(rng.integers(0, len(snap.user_ids)))]
        placement = snap.placements[int(rng.integers(0, len(snap.placements)))]
        cand_rows = rng.choice(len(snap.ad_ids), size=pool, replace=False)
        reqs.append(
            {
                "request_id": f"eval{i:07d}",
                "website_id": wid,
                "user_id": uid,
                "placement": placement,
                "candidate_ad_ids": [snap.ad_ids[r] for r in cand_rows],
            }
        )
    return reqs


def _expected_ctr(requests: list[dict], choose, latent: LatentClickModel) -> float:
    total = 0.0
    for r in requests:
        ad_id = choose(r)
        total += latent.p_click(r["user_id"], ad_id, r["website_id"], r["placement"])
    return total / max(1, len(requests))


def evaluate(
    cfg: Config, snap: FeatureSnapshot, scorer: BidScorer, latent: LatentClickModel,
    n_requests: int | None = None,
) -> dict:
    rng = np.random.default_rng(cfg.seed + 1)
    n = n_requests if n_requests is not None else cfg.sim.n_eval_requests
    reqs = make_eval_requests(cfg, snap, rng, n, cfg.sim.candidate_pool)

    def learned(r: dict) -> str:
        return scorer.score(r["website_id"], r["user_id"], r["placement"],
                            candidate_ad_ids=r["candidate_ad_ids"]).ad_id

    def behavior(r: dict) -> str:
        return r["candidate_ad_ids"][int(rng.integers(0, len(r["candidate_ad_ids"])))]

    def oracle(r: dict) -> str:
        ps = [latent.p_click(r["user_id"], a, r["website_id"], r["placement"])
              for a in r["candidate_ad_ids"]]
        return r["candidate_ad_ids"][int(np.argmax(ps))]

    learned_ctr = _expected_ctr(reqs, learned, latent)
    behavior_ctr = _expected_ctr(reqs, behavior, latent)
    oracle_ctr = _expected_ctr(reqs, oracle, latent)
    uplift = (learned_ctr - behavior_ctr) / behavior_ctr * 100.0 if behavior_ctr > 0 else 0.0
    gap_closed = (
        (learned_ctr - behavior_ctr) / (oracle_ctr - behavior_ctr) * 100.0
        if oracle_ctr > behavior_ctr else 0.0
    )
    result = {
        "n_requests": n,
        "learned_ctr": learned_ctr,
        "behavior_ctr": behavior_ctr,
        "oracle_ctr": oracle_ctr,
        "uplift_pct": uplift,
        "oracle_gap_closed_pct": gap_closed,
        "snips_logged_uplift_pct": _snips_uplift(cfg, snap, scorer),
    }
    return result


def _snips_uplift(cfg: Config, snap: FeatureSnapshot, scorer: BidScorer) -> float | None:
    """Self-normalized IPS estimate of the learned policy's click rate on *logged won* data,
    relative to the logged click rate. Behavior selection is ~uniform over the candidate set,
    so we approximate the per-impression propensity as 1/|known ads|; the target policy is the
    deterministic argmax over the same candidate set restricted to the logged ad. This is a
    coarse, model-free sanity check (high variance), not the headline metric."""
    try:
        logs = loaders.load_bid_logs_df(cfg).filter(pl.col("won"))
    except Exception:  # noqa: BLE001
        return None
    if logs.height == 0:
        return None
    sample = logs.sample(n=min(2000, logs.height), seed=cfg.seed)
    num = den = base = 0.0
    n_ads = len(snap.ad_ids)
    for row in sample.iter_rows(named=True):
        res = scorer.score(row["website_id"], row["user_id"], row["placement"])
        # target picks argmax over ALL known ads; reward credited only if it matches the log.
        w = float(n_ads) if res.ad_id == row["ad_id"] else 0.0
        num += w * float(row["click"])
        den += w
        base += float(row["click"])
    if den == 0:
        return None
    learned = num / den
    logged = base / sample.height
    return (learned - logged) / logged * 100.0 if logged > 0 else None


def coldstart_demo(cfg: Config, snap: FeatureSnapshot, scorer: BidScorer) -> dict | None:
    """Score a held-out cold-start ad against known ads on a category-matching request."""
    ads = loaders.load_ads(cfg)
    cold = [a for a in ads if a.is_coldstart]
    if not cold:
        return None
    embedder = get_embedder(cfg)
    target = cold[0]
    emb = embedder.embed([target.content_text()])[0]
    cold_ad = ColdAd(ad_id=target.ad_id, content_emb=emb, bid_cap_jpy=target.bid_cap_jpy)

    # pick a website in the cold ad's category and a user who engages that category
    sites = [w for w, v in zip(snap.website_ids, snap.website_vertical, strict=True)
             if v == target.category]
    website_id = sites[0] if sites else snap.website_ids[0]
    rng = np.random.default_rng(cfg.seed + 7)
    cand = [snap.ad_ids[r] for r in rng.choice(len(snap.ad_ids),
            size=min(cfg.sim.candidate_pool, len(snap.ad_ids)), replace=False)]

    res = scorer.score(website_id, snap.user_ids[0], "header",
                       candidate_ad_ids=cand, cold_candidates=[cold_ad])
    ranking = sorted(res.q_by_ad.items(), key=lambda kv: -kv[1])
    rank = [aid for aid, _ in ranking].index(target.ad_id) + 1
    return {
        "cold_ad_id": target.ad_id,
        "category": target.category,
        "website_id": website_id,
        "cold_ad_q": res.q_by_ad[target.ad_id],
        "rank_among_candidates": rank,
        "n_candidates": len(ranking),
        "chosen_ad_id": res.ad_id,
        "chosen_is_cold": res.ad_id == target.ad_id,
    }
