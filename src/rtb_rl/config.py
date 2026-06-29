"""Typed configuration loaded from ``configs/config.yaml`` with environment overrides.

Loading precedence (lowest to highest):
1. defaults baked into the pydantic models below,
2. ``configs/config.yaml`` (path overridable via ``RTB_CONFIG_FILE``),
3. environment variables (``RTB__SECTION__FIELD`` for model fields; plus the
   ``POSTGRES_DSN`` / ``REDIS_URL`` secrets which are read directly).

The embedding *dimensionality* is intentionally NOT stored here — it is reported by the
active :class:`~rtb_rl.embeddings.base.Embedder` at runtime so the feature store and the
Q-network always agree with whatever provider is actually in use.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = REPO_ROOT / "configs" / "config.yaml"


class PathsConfig(BaseModel):
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"


class DataConfig(BaseModel):
    n_websites: int = 60
    n_users: int = 2000
    n_campaigns: int = 20
    n_ads: int = 120
    n_bid_logs: int = 50_000
    n_coldstart_ads: int = 8
    placements: list[str] = Field(
        default_factory=lambda: ["header", "sidebar", "in_article", "footer", "interstitial"]
    )
    verticals: list[str] = Field(
        default_factory=lambda: [
            "finance", "ecommerce", "news", "gaming", "travel",
            "health", "tech", "food", "sports", "entertainment",
        ]
    )


class EmbeddingsConfig(BaseModel):
    provider: str = "local"  # local | api | hashing
    model_name: str = "intfloat/multilingual-e5-small"
    hashing_dim: int = 256
    api_provider: str = "vertex"
    api_model: str = "text-embedding-3-small"
    batch_size: int = 64


class FeaturesConfig(BaseModel):
    affinity_top_k: int = 50


class RLConfig(BaseModel):
    gamma: float = 0.85
    lr: float = 5e-4
    hidden_dim: int = 256
    batch_size: int = 256
    epochs: int = 5
    grad_steps_per_epoch: int = 400
    target_update_tau: float = 5e-3
    cql_alpha: float = 1.0
    double_dqn: bool = True
    dueling: bool = True
    click_reward: float = 1.0
    cost_coef: float = 5e-5


class SimConfig(BaseModel):
    n_eval_requests: int = 5000
    candidate_pool: int = 40


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    candidate_pool: int = 40
    default_floor_jpy: float = 30.0


class RetrainConfig(BaseModel):
    interval_hours: float = 6
    recent_window_hours: float = 24
    uplift_gate: float = 0.0


class StoreConfig(BaseModel):
    feature_backend: str = "memory"  # memory | sqlite | postgres
    cache_backend: str = "memory"  # memory (fakeredis) | redis
    sqlite_path: str = "data/processed/features.db"
    # Secrets — populated from the environment, never committed to YAML.
    postgres_dsn: str | None = None
    redis_url: str | None = None


class Config(BaseSettings):
    """Root configuration object. Access via :func:`get_config`."""

    model_config = SettingsConfigDict(env_prefix="RTB__", env_nested_delimiter="__")

    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    retrain: RetrainConfig = Field(default_factory=RetrainConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)

    # ---- convenience: absolute, ensured-to-exist directories ----
    def abspath(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

    def ensure_dirs(self) -> None:
        for p in (
            self.paths.raw_dir,
            self.paths.processed_dir,
            self.paths.checkpoints_dir,
            self.paths.results_dir,
        ):
            self.abspath(p).mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the process-wide configuration (cached)."""
    cfg_file = Path(os.environ.get("RTB_CONFIG_FILE", DEFAULT_CONFIG_FILE))
    data = _load_yaml(cfg_file)
    cfg = Config(**data)
    # Secrets come from the environment regardless of YAML contents.
    cfg.store.postgres_dsn = os.environ.get("POSTGRES_DSN", cfg.store.postgres_dsn)
    cfg.store.redis_url = os.environ.get("REDIS_URL", cfg.store.redis_url)
    return cfg


def reset_config_cache() -> None:
    """Clear the cached config (useful in tests that tweak the environment)."""
    get_config.cache_clear()
