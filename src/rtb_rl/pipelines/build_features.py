"""Offline feature-build pipeline: embeddings -> affinity -> snapshot (+ optional SQL/Redis).

Reads the raw synthetic dataset, computes website/ad content embeddings, engagement-weighted
user embeddings, per-(site, placement) market context, and the top-K affinity table; then
writes the portable :class:`FeatureSnapshot`. When ``store.feature_backend``/``cache_backend``
are not ``memory`` it also populates the durable SQL store and warms the Redis cache.
"""

from __future__ import annotations

import asyncio
import logging

import polars as pl

from rtb_rl.config import Config, get_config
from rtb_rl.data import loaders
from rtb_rl.embeddings import get_embedder
from rtb_rl.features import affinity as affinity_mod
from rtb_rl.features.store import FeatureSnapshot, make_cache, make_durable_store
from rtb_rl.features.user import build_user_features
from rtb_rl.features.website import build_ad_features, build_website_features

logger = logging.getLogger(__name__)


def _market_stats(bid_logs: pl.DataFrame) -> dict[str, list[float]]:
    grp = bid_logs.group_by(["website_id", "placement"]).agg(
        win_rate=pl.col("won").mean(),
        avg_clear=pl.col("market_price_jpy").mean(),
    )
    out: dict[str, list[float]] = {}
    for row in grp.iter_rows(named=True):
        out[f"{row['website_id']}|{row['placement']}"] = [
            float(row["win_rate"]),
            float(row["avg_clear"]),
        ]
    return out


def build_features(cfg: Config | None = None) -> FeatureSnapshot:
    cfg = cfg or get_config()
    cfg.ensure_dirs()
    ds = loaders.load_dataset(cfg)
    bid_logs = loaders.load_bid_logs_df(cfg)
    embedder = get_embedder(cfg)
    logger.info("Embedding with provider=%s dim=%d", embedder.name, embedder.dim)

    wf = build_website_features(embedder, ds.websites)
    uf = build_user_features(ds.users, wf["ids"], wf["emb"])
    af = build_ad_features(embedder, ds.ads, bid_logs)

    snap = FeatureSnapshot(
        embed_dim=embedder.dim,
        placements=cfg.data.placements,
        verticals=cfg.data.verticals,
        website_ids=wf["ids"],
        website_emb=wf["emb"],
        website_base_ctr=wf["base_ctr"],
        website_base_cvr=wf["base_cvr"],
        website_vertical=wf["vertical"],
        user_ids=uf["ids"],
        user_emb=uf["emb"],
        ad_ids=af["ids"],
        ad_emb=af["emb"],
        ad_bid_cap=af["bid_cap"],
        ad_smoothed_ctr=af["smoothed_ctr"],
        ad_is_cold=af["is_cold"],
        ad_category=af["category"],
        ad_targets=af["targets"],
        market=_market_stats(bid_logs),
    )
    snap.save(cfg)

    # Offline affinity batch job (top-K per user) -> parquet artifact.
    aff = affinity_mod.compute_topk_affinity(
        uf["ids"], uf["emb"], wf["ids"], wf["emb"], cfg.features.affinity_top_k
    )
    affinity_mod.save_affinity(cfg, aff)

    _populate_production_sinks(cfg, snap, aff)

    logger.info(
        "Features built: %d websites, %d users, %d known ads, %d affinity rows",
        len(snap.website_ids), len(snap.user_ids), len(snap.ad_ids), aff.height,
    )
    return snap


def _populate_production_sinks(cfg: Config, snap: FeatureSnapshot, aff: pl.DataFrame) -> None:
    store = make_durable_store(cfg)
    if store is not None:
        for i, wid in enumerate(snap.website_ids):
            store.put("website", wid, {"base_ctr": float(snap.website_base_ctr[i]),
                                       "base_cvr": float(snap.website_base_cvr[i]),
                                       "vertical": snap.website_vertical[i]}, snap.website_emb[i])
        for i, uid in enumerate(snap.user_ids):
            store.put("user", uid, None, snap.user_emb[i])
        for i, aid in enumerate(snap.ad_ids):
            store.put("ad", aid, {"smoothed_ctr": float(snap.ad_smoothed_ctr[i]),
                                  "bid_cap": float(snap.ad_bid_cap[i])}, snap.ad_emb[i])
        logger.info("Durable feature store populated (%s).", cfg.store.feature_backend)

    if cfg.store.cache_backend == "redis":
        asyncio.run(_warm_cache(cfg, snap, aff))


async def _warm_cache(cfg: Config, snap: FeatureSnapshot, aff: pl.DataFrame) -> None:
    cache = make_cache(cfg)
    try:
        for i, uid in enumerate(snap.user_ids):
            await cache.set_vector(f"user:{uid}", snap.user_emb[i])
        for i, wid in enumerate(snap.website_ids):
            await cache.set_vector(f"website:{wid}", snap.website_emb[i])
        for row in aff.iter_rows(named=True):
            await cache.set_float(f"aff:{row['user_id']}:{row['website_id']}", row["affinity"])
        logger.info("Redis cache warmed.")
    finally:
        await cache.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_features()
