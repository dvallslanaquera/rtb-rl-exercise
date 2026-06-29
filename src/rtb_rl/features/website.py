"""Content embeddings for websites and ads, plus the ad's Bayesian-smoothed CTR prior."""

from __future__ import annotations

import numpy as np
import polars as pl

from rtb_rl.embeddings.base import Embedder
from rtb_rl.schemas import Ad, Website


def build_website_features(embedder: Embedder, websites: list[Website]) -> dict:
    emb = embedder.embed([w.content_text() for w in websites])
    return {
        "ids": [w.website_id for w in websites],
        "emb": emb,
        "base_ctr": np.array([w.base_ctr for w in websites], dtype=np.float32),
        "base_cvr": np.array([w.base_cvr for w in websites], dtype=np.float32),
        "vertical": [w.vertical for w in websites],
    }


def _smoothed_ctr(bid_logs: pl.DataFrame, ad_ids: list[str], prior_ctr: float = 0.02,
                  strength: float = 50.0) -> dict[str, float]:
    """Empirical-Bayes CTR per ad: (clicks + a) / (wins + a + b), shrinking sparse ads toward
    the global prior. Computed only over WON impressions (the only ones where a click is
    observable)."""
    won = bid_logs.filter(pl.col("won"))
    grp = (
        won.group_by("ad_id")
        .agg(wins=pl.len(), clicks=pl.col("click").sum())
        if won.height
        else pl.DataFrame({"ad_id": [], "wins": [], "clicks": []})
    )
    stats = {row["ad_id"]: (row["wins"], row["clicks"]) for row in grp.iter_rows(named=True)}
    a = prior_ctr * strength
    b = (1.0 - prior_ctr) * strength
    out: dict[str, float] = {}
    for aid in ad_ids:
        wins, clicks = stats.get(aid, (0, 0))
        out[aid] = float((clicks + a) / (wins + a + b))
    return out


def build_ad_features(embedder: Embedder, ads: list[Ad], bid_logs: pl.DataFrame) -> dict:
    """Build features for the *known* ads (cold-start ads are excluded here — they have no
    history and are handled by :mod:`rtb_rl.rl.cold_start`)."""
    known = [a for a in ads if not a.is_coldstart]
    ctr = _smoothed_ctr(bid_logs, [a.ad_id for a in known])
    emb = embedder.embed([a.content_text() for a in known])
    return {
        "ids": [a.ad_id for a in known],
        "emb": emb,
        "bid_cap": np.array([a.bid_cap_jpy for a in known], dtype=np.float32),
        "smoothed_ctr": np.array([ctr[a.ad_id] for a in known], dtype=np.float32),
        "is_cold": np.array([False] * len(known), dtype=bool),
        "category": [a.category for a in known],
        "targets": [a.target_verticals for a in known],
    }
