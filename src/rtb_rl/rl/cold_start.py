"""Cold-start priors via nearest-neighbor borrowing in embedding space.

New ad / campaign (no history): find the K most similar *known* ads by content embedding and
borrow (a) a similarity-weighted average of their learned id-embeddings and (b) a CTR prior.
Because the Q-model scores content features + id-embedding, a brand-new creative is then scored
sensibly on its first impression instead of cold-zero.

New user (no engagement -> zero vector): assume alignment with the content currently being
viewed (use the website embedding as the user prior). This is a deliberately simple,
defensible prior; it decays automatically as the user accrues real engagement.

The empirical-Bayes shrinkage that blends a borrowed prior with accruing real data lives in the
smoothed-CTR computation (:func:`rtb_rl.features.website._smoothed_ctr`); here we surface the
borrowed prior itself.
"""

from __future__ import annotations

import numpy as np

from rtb_rl.features.store import FeatureSnapshot


class ColdStartResolver:
    def __init__(self, snap: FeatureSnapshot, id_table: np.ndarray, k: int = 5) -> None:
        self.snap = snap
        self.id_table = id_table.astype(np.float32)  # (Na, id_dim)
        self.k = min(k, len(snap.ad_ids))
        self._ad_emb = snap.ad_emb  # (Na, E), L2-normalized

    def _neighbors(self, content_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sims = self._ad_emb @ content_emb  # cosine (both normalized)
        idx = np.argpartition(-sims, kth=self.k - 1)[: self.k]
        idx = idx[np.argsort(-sims[idx])]
        weights = np.clip(sims[idx], 0.0, None)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        return idx, weights / weights.sum()

    def id_embedding_for(self, content_emb: np.ndarray) -> np.ndarray:
        """Similarity-weighted average of neighbor id-embeddings for a new ad."""
        idx, w = self._neighbors(content_emb)
        return (self.id_table[idx] * w[:, None]).sum(axis=0).astype(np.float32)

    def smoothed_ctr_for(self, content_emb: np.ndarray) -> float:
        idx, w = self._neighbors(content_emb)
        return float((self.snap.ad_smoothed_ctr[idx] * w).sum())

    def user_vector(self, uid: str, website_id: str) -> tuple[np.ndarray, bool]:
        """Return (user_vec, is_cold). Falls back to the website embedding for unknown or
        zero-engagement users."""
        if self.snap.has_user(uid):
            vec = self.snap.user_vec(uid)
            if np.linalg.norm(vec) > 1e-6:
                return vec, False
        if self.snap.has_website(website_id):
            return self.snap.website_vec(website_id), True
        return np.zeros(self.snap.embed_dim, dtype=np.float32), True
