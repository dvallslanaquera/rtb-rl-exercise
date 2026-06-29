"""API-backed embedder (Vertex AI or OpenAI via LangChain).

Used only when ``embeddings.provider == "api"``. Requires credentials (see .env.example) and
network access; imports are lazy so the dependency is optional.
"""

from __future__ import annotations

import numpy as np

from rtb_rl.embeddings.base import Embedder, l2_normalize


class ApiEmbedder(Embedder):
    name = "api"

    def __init__(self, provider: str = "vertex", model: str = "text-embedding-3-small") -> None:
        self._provider = provider
        self._model = model
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            self._client = OpenAIEmbeddings(model=model)
        elif provider == "vertex":
            # Requires GOOGLE_APPLICATION_CREDENTIALS / a configured GCP project.
            from langchain_google_vertexai import VertexAIEmbeddings

            self._client = VertexAIEmbeddings(model_name=model)
        else:
            raise ValueError(f"Unknown api embedding provider: {provider!r}")
        self._dim = len(self._client.embed_query("dimension probe"))

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._client.embed_documents(list(texts))
        return l2_normalize(np.asarray(vectors, dtype=np.float32))
