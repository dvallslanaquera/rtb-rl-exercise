"""Serving-side view of the distributed feature cache.

The hot path reads features from an in-process :class:`FeatureSnapshot` (dict lookup + a tiny
batched matmul → sub-millisecond), which in production is what each stateless replica loads
from the shared Redis cache and what the retraining loop refreshes. This module simply exposes
the cache wiring so serving can report/refresh it; see :mod:`rtb_rl.features.store` for the
implementation.
"""

from __future__ import annotations

from rtb_rl.features.store import FeatureCache, make_cache

__all__ = ["FeatureCache", "make_cache"]
