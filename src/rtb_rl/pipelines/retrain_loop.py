"""Every-N-hours retraining loop.

Each cycle: refresh features from the recent log window → warm-start retrain the DQN from the
current production model → evaluate the candidate in the offline simulator → **promote only if
the CTR uplift clears the gate**, otherwise keep the incumbent. Promotion flips the registry
pointer, which the serving layer hot-swaps to. This is what lets the policy track market drift
(new competitor campaigns, budget/pacing changes) without a human in the loop.

Run continuously with ``rtb retrain`` (APScheduler) or a single cycle with ``rtb retrain --once``.
In production this DAG maps onto Vertex AI Pipelines (see ``infra/vertex/pipeline.py``).
"""

from __future__ import annotations

import logging

from rtb_rl.config import Config, get_config
from rtb_rl.data import loaders
from rtb_rl.features.store import FeatureSnapshot
from rtb_rl.pipelines.build_features import build_features
from rtb_rl.registry import ModelRegistry
from rtb_rl.rl.trainer import train as train_agent
from rtb_rl.serving.inference import BidScorer
from rtb_rl.sim import evaluate as sim_eval

logger = logging.getLogger(__name__)


def run_once(cfg: Config | None = None) -> dict:
    cfg = cfg or get_config()
    registry = ModelRegistry(cfg)
    prev_version = registry.latest_version()

    # 1) refresh features (embeddings + affinity + market context) from the latest logs
    snap = build_features(cfg)

    # 2) warm-start retrain from the incumbent, register WITHOUT promoting yet
    agent, meta = train_agent(
        cfg, snap=snap, warm_start_version=prev_version, register=True, promote=False
    )
    assert meta is not None

    # 3) gate on offline simulator CTR uplift before promotion
    latent = loaders.load_latent(cfg)
    scorer = BidScorer(snap, agent.online, meta, cfg)
    result = sim_eval.evaluate(cfg, snap, scorer, latent)
    uplift = result["uplift_pct"]

    promoted = uplift >= cfg.retrain.uplift_gate
    if promoted:
        registry.promote(meta.version)
        logger.info("Promoted %s (uplift %.2f%% >= gate %.2f%%)",
                    meta.version, uplift, cfg.retrain.uplift_gate)
    else:
        logger.info("Rejected %s (uplift %.2f%% < gate %.2f%%); keeping %s",
                    meta.version, uplift, cfg.retrain.uplift_gate, prev_version)

    return {
        "candidate_version": meta.version,
        "previous_version": prev_version,
        "promoted": promoted,
        "active_version": registry.latest_version(),
        **result,
    }


def run_loop(cfg: Config | None = None) -> None:
    """Block and run :func:`run_once` every ``retrain.interval_hours`` (APScheduler)."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    cfg = cfg or get_config()
    if not FeatureSnapshot.exists(cfg):
        build_features(cfg)
    logger.info("Starting retrain loop: every %.2f h", cfg.retrain.interval_hours)
    run_once(cfg)  # run immediately, then on schedule
    scheduler = BlockingScheduler()
    scheduler.add_job(
        lambda: run_once(cfg),
        "interval",
        hours=cfg.retrain.interval_hours,
        id="retrain",
        max_instances=1,
        coalesce=True,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Retrain loop stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_loop()
