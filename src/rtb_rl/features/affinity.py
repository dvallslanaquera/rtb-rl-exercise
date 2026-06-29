"""Offline user↔website affinity batch job.

Affinity = cosine similarity of the (L2-normalized) user and website embeddings. We compute
the full user×website score matrix in chunks and materialize the **top-K websites per user**
to ``data/processed/affinity.parquet`` — the offline artifact a production system would push
into Redis for the serving hot path and reuse for cold-start neighbor lookups.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from rtb_rl.config import Config


def compute_topk_affinity(
    user_ids: list[str],
    user_emb: np.ndarray,
    website_ids: list[str],
    website_emb: np.ndarray,
    top_k: int,
    chunk: int = 1024,
) -> pl.DataFrame:
    website_ids_arr = np.array(website_ids)
    k = min(top_k, len(website_ids))
    rows_u: list[str] = []
    rows_w: list[str] = []
    rows_a: list[float] = []
    wt = website_emb.T  # (E, Nw)
    for start in range(0, len(user_ids), chunk):
        block = user_emb[start : start + chunk]  # (b, E)
        scores = block @ wt  # (b, Nw) cosine, since both are L2-normalized
        # top-k indices per row (unsorted partition then sort within the k)
        idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        for r in range(block.shape[0]):
            cols = idx[r]
            order = np.argsort(-scores[r, cols])
            cols = cols[order]
            uid = user_ids[start + r]
            rows_u.extend([uid] * k)
            rows_w.extend(website_ids_arr[cols].tolist())
            rows_a.extend(scores[r, cols].astype(float).tolist())
    return pl.DataFrame({"user_id": rows_u, "website_id": rows_w, "affinity": rows_a})


def save_affinity(cfg: Config, df: pl.DataFrame) -> None:
    proc = cfg.abspath(cfg.paths.processed_dir)
    proc.mkdir(parents=True, exist_ok=True)
    df.write_parquet(proc / "affinity.parquet")


def load_affinity(cfg: Config) -> pl.DataFrame:
    return pl.read_parquet(cfg.abspath(cfg.paths.processed_dir) / "affinity.parquet")
