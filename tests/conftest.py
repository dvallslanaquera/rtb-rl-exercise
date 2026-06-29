"""Shared pytest fixtures.

All tests run fully offline: the deterministic hashing embedder + in-memory store, a tiny
synthetic dataset, and a registry isolated to a temp dir. No network, no Postgres/Redis, no
model download.
"""

from __future__ import annotations

import pytest

from rtb_rl.config import Config


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path_factory, monkeypatch):
    reg = tmp_path_factory.mktemp("registry")
    monkeypatch.setenv("RTB_REGISTRY_DIR", str(reg))
    yield


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config(
        seed=0,
        paths={
            "data_dir": str(tmp_path / "data"),
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "checkpoints_dir": str(tmp_path / "checkpoints"),
            "results_dir": str(tmp_path / "results"),
        },
        data={
            "n_websites": 12,
            "n_users": 80,
            "n_campaigns": 5,
            "n_ads": 20,
            "n_bid_logs": 2500,
            "n_coldstart_ads": 3,
        },
        embeddings={"provider": "hashing", "hashing_dim": 64},
        features={"affinity_top_k": 5},
        rl={
            "epochs": 1,
            "grad_steps_per_epoch": 40,
            "batch_size": 64,
            "hidden_dim": 64,
            "cql_alpha": 1.0,
        },
        sim={"n_eval_requests": 300, "candidate_pool": 8},
        store={"feature_backend": "memory", "cache_backend": "memory"},
    )
    c.ensure_dirs()
    return c
