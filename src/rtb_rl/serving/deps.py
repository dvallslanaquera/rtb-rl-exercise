"""Serving state: holds the warm in-memory model + feature snapshot + scorer, and performs
model **hot-swap** by polling the registry's latest pointer.

This is the hand-off from the every-N-hours retraining loop: when the loop promotes a new model
version, the background poller here reloads it without a restart and serving traffic moves to
the new policy.
"""

from __future__ import annotations

import logging
import threading

from rtb_rl.config import Config, get_config
from rtb_rl.features.store import FeatureSnapshot, make_cache
from rtb_rl.registry import ModelRegistry
from rtb_rl.serving.inference import BidScorer

logger = logging.getLogger(__name__)
HOT_SWAP_POLL_SECONDS = 5.0


class ServingState:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or get_config()
        self.registry = ModelRegistry(self.cfg)
        self.cache = make_cache(self.cfg)
        self._lock = threading.RLock()
        self.snapshot: FeatureSnapshot | None = None
        self.scorer: BidScorer | None = None
        self.version: str | None = None

    def ready(self) -> bool:
        return self.scorer is not None

    def load(self) -> None:
        """Load the feature snapshot + latest registered model (idempotent)."""
        with self._lock:
            if self.snapshot is None:
                if not FeatureSnapshot.exists(self.cfg):
                    raise FileNotFoundError(
                        "Feature snapshot missing — run `rtb build-features` first."
                    )
                self.snapshot = FeatureSnapshot.load(self.cfg)
            model, meta = self.registry.load_latest()
            self.scorer = BidScorer(self.snapshot, model, meta, self.cfg)
            self.version = meta.version
            logger.info("Loaded model %s (%d known ads)", meta.version, meta.n_ads)

    def maybe_hot_swap(self) -> bool:
        """Reload if the registry's latest version differs from the one in memory."""
        latest = self.registry.latest_version()
        if latest is not None and latest != self.version:
            logger.info("Hot-swapping model %s -> %s", self.version, latest)
            self.load()
            return True
        return False


_state: ServingState | None = None


def get_state() -> ServingState:
    global _state
    if _state is None:
        _state = ServingState()
    return _state
