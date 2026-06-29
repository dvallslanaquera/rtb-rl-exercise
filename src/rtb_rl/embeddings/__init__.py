"""Pluggable text-embedding providers (local / api / hashing)."""

from rtb_rl.embeddings.base import Embedder
from rtb_rl.embeddings.factory import get_embedder

__all__ = ["Embedder", "get_embedder"]
