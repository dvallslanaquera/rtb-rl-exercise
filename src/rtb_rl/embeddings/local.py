"""Local multilingual embedder (sentence-transformers via LangChain).

Default provider. Imports are lazy and the heavy deps live in the ``[embeddings]`` extra; the
:mod:`~rtb_rl.embeddings.factory` falls back to :class:`~rtb_rl.embeddings.hashing.HashingEmbedder`
if the model (or its dependencies) cannot be loaded — so the project always runs.
"""

from __future__ import annotations

import numpy as np

from rtb_rl.embeddings.base import Embedder, l2_normalize


class LocalEmbedder(Embedder):
    name = "local"

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        # langchain-huggingface wraps sentence-transformers and exposes a stable interface.
        from langchain_huggingface import HuggingFaceEmbeddings

        self._model_name = model_name
        self._client = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        # Probe dimensionality once.
        self._dim = len(self._client.embed_query("dimension probe"))

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        # e5 models expect a "passage:" / "query:" prefix; "passage:" suits document content.
        vectors = self._client.embed_documents([f"passage: {t}" for t in texts])
        return l2_normalize(np.asarray(vectors, dtype=np.float32))
