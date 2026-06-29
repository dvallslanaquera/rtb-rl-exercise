"""Embedder interface shared by the local, API and hashing providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Maps text -> L2-normalized vectors. Implementations must be deterministic so that
    embeddings computed offline (training) match those recomputed at serving time."""

    name: str = "base"

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array of L2-normalized embeddings."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (mat / norms).astype(np.float32)
