"""Thin pipeline wrapper around :func:`rtb_rl.rl.trainer.train`."""

from __future__ import annotations

import logging

from rtb_rl.config import Config, get_config
from rtb_rl.registry import ModelMeta
from rtb_rl.rl.trainer import train as _train

logger = logging.getLogger(__name__)


def train(cfg: Config | None = None, warm_start_version: str | None = None) -> ModelMeta | None:
    cfg = cfg or get_config()
    _agent, meta = _train(cfg, warm_start_version=warm_start_version)
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
