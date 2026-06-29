"""Deterministic feature-hashing embedder — the always-available offline/CI fallback.

Uses character n-grams (Japanese is not whitespace-delimited) plus whitespace tokens, hashed
into a fixed-width signed vector via MD5 (stable across processes, unlike Python's salted
``hash``). Same-vertical templates share n-grams, so cosine similarity still encodes topical
relatedness — enough to drive user↔website affinity and vertical matching without any model
download or network access.
"""

from __future__ import annotations

import hashlib

import numpy as np

from rtb_rl.embeddings.base import Embedder, l2_normalize

_CHAR_NGRAMS = (2, 3)


def _tokens(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    toks: list[str] = [f"w:{t}" for t in text.split()]
    compact = text.replace(" ", "")
    for n in _CHAR_NGRAMS:
        if len(compact) >= n:
            toks.extend(f"c{n}:{compact[i : i + n]}" for i in range(len(compact) - n + 1))
    return toks


def _hash(token: str) -> tuple[int, float]:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "little")
    sign = 1.0 if digest[8] & 1 else -1.0
    return bucket, sign


class HashingEmbedder(Embedder):
    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _tokens(text):
                bucket, sign = _hash(tok)
                out[i, bucket % self._dim] += sign
        return l2_normalize(out)
