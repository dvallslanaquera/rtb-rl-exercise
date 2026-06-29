"""Config-driven embedder selection with graceful fallback.

``provider == "local"`` is the default; if the local model (or its heavy optional deps) can't
be loaded, we log a warning and fall back to the deterministic hashing embedder so the rest of
the pipeline keeps working offline. ``provider == "hashing"`` forces the fallback (used in CI
and tests); ``provider == "api"`` uses a hosted model and does *not* silently fall back.
"""

from __future__ import annotations

import logging

from rtb_rl.config import Config, get_config
from rtb_rl.embeddings.base import Embedder
from rtb_rl.embeddings.hashing import HashingEmbedder

logger = logging.getLogger(__name__)


def get_embedder(cfg: Config | None = None) -> Embedder:
    cfg = cfg or get_config()
    ec = cfg.embeddings
    provider = ec.provider.lower()

    if provider == "hashing":
        return HashingEmbedder(dim=ec.hashing_dim)

    if provider == "api":
        from rtb_rl.embeddings.api import ApiEmbedder

        return ApiEmbedder(provider=ec.api_provider, model=ec.api_model)

    if provider == "local":
        try:
            from rtb_rl.embeddings.local import LocalEmbedder

            return LocalEmbedder(model_name=ec.model_name)
        except Exception as exc:  # noqa: BLE001 - any load/import failure -> safe fallback
            logger.warning(
                "Local embedder '%s' unavailable (%s); falling back to HashingEmbedder(dim=%d). "
                "Install the [embeddings] extra for real multilingual embeddings.",
                ec.model_name,
                exc,
                ec.hashing_dim,
            )
            return HashingEmbedder(dim=ec.hashing_dim)

    raise ValueError(f"Unknown embeddings.provider: {ec.provider!r}")
