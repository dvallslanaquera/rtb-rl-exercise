"""Feature store + cache.

Three layers, all writing/reading the *same* precomputed features:

- :class:`FeatureSnapshot` — a portable in-process view (numpy arrays + id->row indices),
  serialized to ``data/processed/feature_snapshot.npz`` (+ json sidecar). This is the
  workhorse used by training, simulation and serving; it needs no external services and
  guarantees the demo runs cross-process.
- :class:`DurableFeatureStore` — a SQLAlchemy key/value store (sqlite or postgres), the
  "source of truth" in the production narrative. Populated by ``build_features`` when
  ``store.feature_backend != memory``.
- :class:`FeatureCache` — an async key/value cache (redis or an in-memory dict) embodying the
  distributed Redis cache on the 10ms serving hot path.

User↔website affinity is cosine similarity of stored embeddings, so the snapshot can compute
it on demand; the offline batch job additionally materializes a top-K table (see
:mod:`rtb_rl.features.affinity`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rtb_rl.config import Config

SNAPSHOT_NPZ = "feature_snapshot.npz"
SNAPSHOT_JSON = "feature_snapshot.json"


def _index(ids: list[str]) -> dict[str, int]:
    return {k: i for i, k in enumerate(ids)}


@dataclass
class FeatureSnapshot:
    """Read-optimized, in-memory view of all precomputed features."""

    embed_dim: int
    placements: list[str]
    verticals: list[str]

    website_ids: list[str]
    website_emb: np.ndarray  # (Nw, E)
    website_base_ctr: np.ndarray  # (Nw,)
    website_base_cvr: np.ndarray  # (Nw,)
    website_vertical: list[str]

    user_ids: list[str]
    user_emb: np.ndarray  # (Nu, E)

    ad_ids: list[str]  # known ads, in id-embedding-table order
    ad_emb: np.ndarray  # (Na, E)
    ad_bid_cap: np.ndarray  # (Na,)
    ad_smoothed_ctr: np.ndarray  # (Na,)
    ad_is_cold: np.ndarray  # (Na,) bool
    ad_category: list[str]
    ad_targets: list[list[str]]

    # market context per (website_id, placement) -> [win_rate, avg_clearing_jpy]
    market: dict[str, list[float]] = field(default_factory=dict)

    # lazily built indices
    _w_idx: dict[str, int] = field(default_factory=dict, repr=False)
    _u_idx: dict[str, int] = field(default_factory=dict, repr=False)
    _a_idx: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._w_idx = _index(self.website_ids)
        self._u_idx = _index(self.user_ids)
        self._a_idx = _index(self.ad_ids)

    # ---- accessors ----
    def has_user(self, uid: str) -> bool:
        return uid in self._u_idx

    def has_website(self, wid: str) -> bool:
        return wid in self._w_idx

    def website_vec(self, wid: str) -> np.ndarray:
        return self.website_emb[self._w_idx[wid]]

    def user_vec(self, uid: str) -> np.ndarray:
        return self.user_emb[self._u_idx[uid]]

    def ad_index(self, aid: str) -> int:
        return self._a_idx[aid]

    def affinity(self, uid: str, wid: str) -> float:
        """Cosine affinity between a user and a website (embeddings are L2-normalized)."""
        if uid not in self._u_idx or wid not in self._w_idx:
            return 0.0
        return float(self.user_emb[self._u_idx[uid]] @ self.website_emb[self._w_idx[wid]])

    def market_features(self, wid: str, placement: str) -> tuple[float, float]:
        win_rate, avg_clear = self.market.get(f"{wid}|{placement}", [0.5, 80.0])
        return float(win_rate), float(avg_clear)

    # ---- (de)serialization ----
    def save(self, cfg: Config) -> Path:
        proc = cfg.abspath(cfg.paths.processed_dir)
        proc.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            proc / SNAPSHOT_NPZ,
            website_emb=self.website_emb,
            website_base_ctr=self.website_base_ctr,
            website_base_cvr=self.website_base_cvr,
            user_emb=self.user_emb,
            ad_emb=self.ad_emb,
            ad_bid_cap=self.ad_bid_cap,
            ad_smoothed_ctr=self.ad_smoothed_ctr,
            ad_is_cold=self.ad_is_cold,
        )
        sidecar = {
            "embed_dim": self.embed_dim,
            "placements": self.placements,
            "verticals": self.verticals,
            "website_ids": self.website_ids,
            "website_vertical": self.website_vertical,
            "user_ids": self.user_ids,
            "ad_ids": self.ad_ids,
            "ad_category": self.ad_category,
            "ad_targets": self.ad_targets,
            "market": self.market,
        }
        (proc / SNAPSHOT_JSON).write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")
        return proc / SNAPSHOT_NPZ

    @classmethod
    def load(cls, cfg: Config) -> FeatureSnapshot:
        proc = cfg.abspath(cfg.paths.processed_dir)
        arr = np.load(proc / SNAPSHOT_NPZ, allow_pickle=False)
        meta = json.loads((proc / SNAPSHOT_JSON).read_text(encoding="utf-8"))
        return cls(
            embed_dim=meta["embed_dim"],
            placements=meta["placements"],
            verticals=meta["verticals"],
            website_ids=meta["website_ids"],
            website_emb=arr["website_emb"],
            website_base_ctr=arr["website_base_ctr"],
            website_base_cvr=arr["website_base_cvr"],
            website_vertical=meta["website_vertical"],
            user_ids=meta["user_ids"],
            user_emb=arr["user_emb"],
            ad_ids=meta["ad_ids"],
            ad_emb=arr["ad_emb"],
            ad_bid_cap=arr["ad_bid_cap"],
            ad_smoothed_ctr=arr["ad_smoothed_ctr"],
            ad_is_cold=arr["ad_is_cold"],
            ad_category=meta["ad_category"],
            ad_targets=meta["ad_targets"],
            market=meta["market"],
        )

    @staticmethod
    def exists(cfg: Config) -> bool:
        proc = cfg.abspath(cfg.paths.processed_dir)
        return (proc / SNAPSHOT_NPZ).exists() and (proc / SNAPSHOT_JSON).exists()


# --------------------------------------------------------------------------------------
# Production-path sinks (optional): durable SQL store + async Redis cache.
# --------------------------------------------------------------------------------------
class DurableFeatureStore:
    """SQLAlchemy-backed key/value store (sqlite or postgres). Generic ``(namespace, key)``
    rows hold a JSON payload and/or a float32 vector blob."""

    def __init__(self, url: str) -> None:
        from sqlalchemy import (
            JSON,
            Column,
            LargeBinary,
            MetaData,
            String,
            Table,
            create_engine,
        )

        self.engine = create_engine(url, future=True)
        self.metadata = MetaData()
        self.table = Table(
            "feature_kv",
            self.metadata,
            Column("namespace", String, primary_key=True),
            Column("key", String, primary_key=True),
            Column("payload", JSON, nullable=True),
            Column("vector", LargeBinary, nullable=True),
        )
        self.metadata.create_all(self.engine)

    def put(self, namespace: str, key: str, payload: dict | None, vector: np.ndarray | None) -> None:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        blob = None if vector is None else np.asarray(vector, np.float32).tobytes()
        with self.engine.begin() as conn:
            stmt = sqlite_insert(self.table).values(
                namespace=namespace, key=key, payload=payload, vector=blob
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["namespace", "key"],
                set_={"payload": payload, "vector": blob},
            )
            conn.execute(stmt)

    def get(self, namespace: str, key: str) -> tuple[dict | None, np.ndarray | None]:
        from sqlalchemy import select

        with self.engine.connect() as conn:
            row = conn.execute(
                select(self.table.c.payload, self.table.c.vector).where(
                    self.table.c.namespace == namespace, self.table.c.key == key
                )
            ).first()
        if row is None:
            return None, None
        vec = None if row.vector is None else np.frombuffer(row.vector, dtype=np.float32)
        return row.payload, vec


class FeatureCache:
    """Async key/value cache for the serving hot path. Backed by Redis (production / docker)
    or an in-process dict (default; microsecond reads, zero services)."""

    def __init__(self, backend: str = "memory", redis_url: str | None = None) -> None:
        self.backend = backend
        self._mem: dict[str, bytes] = {}
        self._redis = None
        if backend == "redis":
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(redis_url or "redis://localhost:6379/0")

    async def set_vector(self, key: str, vec: np.ndarray) -> None:
        data = np.asarray(vec, np.float32).tobytes()
        if self._redis is not None:
            await self._redis.set(key, data)
        else:
            self._mem[key] = data

    async def get_vector(self, key: str) -> np.ndarray | None:
        if self._redis is not None:
            data = await self._redis.get(key)
        else:
            data = self._mem.get(key)
        return None if data is None else np.frombuffer(data, dtype=np.float32)

    async def set_float(self, key: str, value: float) -> None:
        await self.set_vector(key, np.asarray([value], np.float32))

    async def get_float(self, key: str) -> float | None:
        vec = await self.get_vector(key)
        return None if vec is None else float(vec[0])

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()


def make_durable_store(cfg: Config) -> DurableFeatureStore | None:
    backend = cfg.store.feature_backend
    if backend == "sqlite":
        path = cfg.abspath(cfg.store.sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return DurableFeatureStore(f"sqlite+pysqlite:///{path}")
    if backend == "postgres":
        dsn = cfg.store.postgres_dsn or "postgresql+psycopg://rtb:rtb@localhost:5432/rtb"
        # Durable store is synchronous -> use the sync driver.
        return DurableFeatureStore(dsn.replace("+asyncpg", "+psycopg"))
    return None  # memory backend uses the FeatureSnapshot only


def make_cache(cfg: Config) -> FeatureCache:
    if cfg.store.cache_backend == "redis":
        return FeatureCache("redis", cfg.store.redis_url)
    return FeatureCache("memory")
