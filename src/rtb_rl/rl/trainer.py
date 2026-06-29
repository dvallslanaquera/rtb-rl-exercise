"""Offline training loop: build the batch dataset, fit the Dueling Double-DQN + CQL agent,
and register the resulting model.

Supports warm-starting from a previous registry version (used by the every-N-hours retrain
loop) when the architecture/inventory is unchanged — so each cycle adapts incrementally rather
than relearning from scratch.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from rtb_rl.config import Config, get_config
from rtb_rl.features.store import FeatureSnapshot
from rtb_rl.registry import ModelMeta, ModelRegistry
from rtb_rl.rl.agent import DQNAgent
from rtb_rl.rl.replay import OfflineDataset, build_offline_dataset

logger = logging.getLogger(__name__)
ID_DIM = 16


def train(
    cfg: Config | None = None,
    snap: FeatureSnapshot | None = None,
    dataset: OfflineDataset | None = None,
    warm_start_version: str | None = None,
    epochs: int | None = None,
    register: bool = True,
    promote: bool = True,
    extra_metrics: dict | None = None,
) -> tuple[DQNAgent, ModelMeta | None]:
    cfg = cfg or get_config()
    snap = snap or FeatureSnapshot.load(cfg)
    ds = dataset or build_offline_dataset(cfg, snap)
    if len(ds) == 0:
        raise RuntimeError("Offline dataset is empty — no won impressions in the logs.")

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    agent = DQNAgent(ds.state_dim, ds.ad_content_dim, ds.n_ads, cfg.rl, id_dim=ID_DIM)

    registry = ModelRegistry(cfg)
    if warm_start_version:
        _maybe_warm_start(agent, registry, warm_start_version)

    n_epochs = epochs if epochs is not None else cfg.rl.epochs
    pool = cfg.sim.candidate_pool
    last: dict = {}
    for epoch in range(n_epochs):
        agg: dict[str, float] = {}
        for _ in range(cfg.rl.grad_steps_per_epoch):
            batch = ds.sample_batch(cfg.rl.batch_size, pool, rng)
            info = agent.update(batch)
            for k, v in info.items():
                agg[k] = agg.get(k, 0.0) + v
        last = {k: v / cfg.rl.grad_steps_per_epoch for k, v in agg.items()}
        logger.info(
            "epoch %d/%d  loss=%.4f td=%.4f cql=%.4f q_mean=%.4f",
            epoch + 1, n_epochs, last["loss"], last["td_loss"], last["cql_loss"], last["q_mean"],
        )

    meta = None
    if register:
        metrics = {**last, **(extra_metrics or {}), "n_transitions": len(ds)}
        meta = registry.register(
            agent.online,
            dict(
                state_dim=ds.state_dim,
                ad_content_dim=ds.ad_content_dim,
                n_ads=ds.n_ads,
                id_dim=ID_DIM,
                hidden_dim=cfg.rl.hidden_dim,
                dueling=cfg.rl.dueling,
                embed_dim=snap.embed_dim,
                ad_ids=snap.ad_ids,
                embedder_name=cfg.embeddings.provider,
                metrics=metrics,
                parent_version=warm_start_version,
            ),
            promote=promote,
        )
        logger.info("Registered model %s (metrics: %s)", meta.version, metrics)
    return agent, meta


def _maybe_warm_start(agent: DQNAgent, registry: ModelRegistry, version: str) -> None:
    try:
        prev, _ = registry.load(version)
        agent.online.load_state_dict(prev.state_dict())
        agent.target.load_state_dict(prev.state_dict())
        logger.info("Warm-started from %s", version)
    except Exception as exc:  # noqa: BLE001 - architecture/inventory changed -> train fresh
        logger.warning("Warm start from %s failed (%s); training from scratch.", version, exc)
