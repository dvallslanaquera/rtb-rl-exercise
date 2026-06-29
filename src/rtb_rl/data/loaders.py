"""Persist / load the synthetic dataset as Parquet (+ latent-model JSON).

Parquet keeps the PoC honest about scale — the loaders stream the same columnar format you
would use for "hundreds of millions" of real delivery logs, just with tiny synthetic volumes.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from rtb_rl.config import Config
from rtb_rl.data.synth import LatentClickModel, SyntheticDataset
from rtb_rl.schemas import Ad, BidLog, User, Website


def _raw_dir(cfg: Config) -> Path:
    p = cfg.abspath(cfg.paths.raw_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_dataset(cfg: Config, ds: SyntheticDataset) -> None:
    raw = _raw_dir(cfg)

    pl.DataFrame([w.model_dump() for w in ds.websites]).write_parquet(raw / "websites.parquet")
    pl.DataFrame([a.model_dump() for a in ds.ads]).write_parquet(raw / "ads.parquet")
    pl.DataFrame([b.model_dump() for b in ds.bid_logs]).write_parquet(raw / "bid_logs.parquet")

    # Engagement is a sparse dict -> store as a JSON string column.
    pl.DataFrame(
        {
            "user_id": [u.user_id for u in ds.users],
            "engagement": [json.dumps(u.engagement) for u in ds.users],
        }
    ).write_parquet(raw / "users.parquet")

    ds.latent.to_json(raw / "latent.json")


def load_websites(cfg: Config) -> list[Website]:
    df = pl.read_parquet(_raw_dir(cfg) / "websites.parquet")
    return [Website(**row) for row in df.iter_rows(named=True)]


def load_ads(cfg: Config) -> list[Ad]:
    df = pl.read_parquet(_raw_dir(cfg) / "ads.parquet")
    return [Ad(**row) for row in df.iter_rows(named=True)]


def load_users(cfg: Config) -> list[User]:
    df = pl.read_parquet(_raw_dir(cfg) / "users.parquet")
    return [
        User(user_id=row["user_id"], engagement=json.loads(row["engagement"]))
        for row in df.iter_rows(named=True)
    ]


def load_bid_logs(cfg: Config) -> list[BidLog]:
    df = pl.read_parquet(_raw_dir(cfg) / "bid_logs.parquet")
    return [BidLog(**row) for row in df.iter_rows(named=True)]


def load_bid_logs_df(cfg: Config) -> pl.DataFrame:
    """Return raw bid logs as a Polars frame (preferred for batch/offline training)."""
    return pl.read_parquet(_raw_dir(cfg) / "bid_logs.parquet")


def load_latent(cfg: Config) -> LatentClickModel:
    return LatentClickModel.from_json(_raw_dir(cfg) / "latent.json")


def load_dataset(cfg: Config) -> SyntheticDataset:
    return SyntheticDataset(
        websites=load_websites(cfg),
        users=load_users(cfg),
        ads=load_ads(cfg),
        bid_logs=load_bid_logs(cfg),
        latent=load_latent(cfg),
    )


def raw_exists(cfg: Config) -> bool:
    raw = cfg.abspath(cfg.paths.raw_dir)
    return all(
        (raw / f).exists()
        for f in ("websites.parquet", "users.parquet", "ads.parquet", "bid_logs.parquet")
    )
