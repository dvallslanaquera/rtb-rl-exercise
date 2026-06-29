"""User embeddings from engagement profiles.

A user's vector is the engagement-weighted mean of the embeddings of the websites they
engaged with (weight = level / 10), L2-normalized. Users with no engagement get a zero vector
and are routed through the cold-start prior at scoring time.
"""

from __future__ import annotations

import numpy as np

from rtb_rl.embeddings.base import l2_normalize
from rtb_rl.schemas import MAX_ENGAGEMENT, User


def build_user_features(
    users: list[User], website_ids: list[str], website_emb: np.ndarray
) -> dict:
    w_idx = {wid: i for i, wid in enumerate(website_ids)}
    dim = website_emb.shape[1]
    emb = np.zeros((len(users), dim), dtype=np.float32)
    for i, u in enumerate(users):
        acc = np.zeros(dim, dtype=np.float32)
        wsum = 0.0
        for wid, level in u.engagement.items():
            j = w_idx.get(wid)
            if j is None or level <= 0:
                continue
            w = level / MAX_ENGAGEMENT
            acc += w * website_emb[j]
            wsum += w
        if wsum > 0:
            acc /= wsum
        emb[i] = acc
    return {"ids": [u.user_id for u in users], "emb": l2_normalize(emb)}
