"""Versioned model registry on the local filesystem.

Each ``register`` writes ``registry/<version>/{model.pt, meta.json}``; ``promote`` flips
``registry/latest.json`` to a version. The serving layer polls the latest pointer and
**hot-swaps** when it changes — the hand-off from the every-N-hours retraining loop. In
production this maps to a GCS bucket + Vertex AI Model Registry (see ``infra/vertex``).

``meta`` carries everything needed to rebuild the network architecture and to map ad ids to
id-embedding rows, so a checkpoint is self-describing.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch

from rtb_rl.config import REPO_ROOT, Config, get_config
from rtb_rl.rl.networks import QModel


@dataclass
class ModelMeta:
    version: str
    state_dim: int
    ad_content_dim: int
    n_ads: int
    id_dim: int
    hidden_dim: int
    dueling: bool
    embed_dim: int
    ad_ids: list[str]  # id-embedding-table order
    embedder_name: str
    created_at: str
    metrics: dict = field(default_factory=dict)
    parent_version: str | None = None


def _registry_dir(cfg: Config) -> Path:
    # RTB_REGISTRY_DIR allows isolating the registry (tests, parallel experiments).
    p = Path(os.environ.get("RTB_REGISTRY_DIR", REPO_ROOT / "registry"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _new_version() -> str:
    return datetime.now(UTC).strftime("v%Y%m%d-%H%M%S-%f")


class ModelRegistry:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or get_config()
        self.root = _registry_dir(self.cfg)

    def register(
        self, model: QModel, meta_kwargs: dict, promote: bool = True
    ) -> ModelMeta:
        version = _new_version()
        vdir = self.root / version
        vdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), vdir / "model.pt")
        meta = ModelMeta(version=version, created_at=datetime.now(UTC).isoformat(),
                         **meta_kwargs)
        (vdir / "meta.json").write_text(json.dumps(asdict(meta), ensure_ascii=False), "utf-8")
        if promote:
            self.promote(version)
        return meta

    def promote(self, version: str) -> None:
        (self.root / "latest.json").write_text(json.dumps({"version": version}), "utf-8")

    def latest_version(self) -> str | None:
        ptr = self.root / "latest.json"
        if not ptr.exists():
            return None
        return json.loads(ptr.read_text("utf-8"))["version"]

    def list_versions(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def load(self, version: str, device: str = "cpu") -> tuple[QModel, ModelMeta]:
        vdir = self.root / version
        meta = ModelMeta(**json.loads((vdir / "meta.json").read_text("utf-8")))
        model = QModel(
            state_dim=meta.state_dim,
            ad_content_dim=meta.ad_content_dim,
            n_ads=meta.n_ads,
            id_dim=meta.id_dim,
            hidden_dim=meta.hidden_dim,
            dueling=meta.dueling,
        )
        model.load_state_dict(torch.load(vdir / "model.pt", map_location=device))
        model.eval()
        return model, meta

    def load_latest(self, device: str = "cpu") -> tuple[QModel, ModelMeta]:
        version = self.latest_version()
        if version is None:
            raise FileNotFoundError("No model has been registered yet. Run training first.")
        return self.load(version, device=device)
