# CLAUDE.md

Guidance for AI coding agents (and new contributors) working in this repository.

## What this project is

`rtb-rl` is a **portfolio reconstruction of a real-time-bidding (RTB) yield-optimization PoC**.
For each bid request it picks the candidate ad with the highest predicted click value using an
offline-trained Dueling Double-DQN with a Conservative Q-Learning (CQL) penalty, serves the
decision from FastAPI under a ~10 ms in-process budget, and retrains on a schedule with a
sim-gated promotion + hot-swap. Everything runs locally on synthetic data; the GCP pieces
(`infra/terraform`, `infra/vertex`) are **documented stubs, not live infrastructure**.

Keep the PoC honest: reported metrics come from a simulator that shares the ground-truth click
model with the data generator, so uplift numbers are sanity signals, not production claims.

## Commands

```bash
# Environment: Python 3.12 ONLY (torch has no 3.13+ wheels). Windows dev box; .venv exists.
.venv/Scripts/python.exe -m pytest -q          # 17 tests, fully offline, ~20 s
.venv/Scripts/python.exe -m ruff check src tests scripts
.venv/Scripts/python.exe -m mypy src

# Pipeline (each step is also a `make` target and a scripts/*.py wrapper)
rtb demo            # generate-data → build-features → train → sim, prints CTR table
rtb generate-data   # synthetic websites/users/ads/bid-logs → data/raw/
rtb build-features  # embeddings + affinity + market context → data/processed/ snapshot
rtb train           # offline DQN+CQL → registry/<version>/, promotes latest
rtb sim             # evaluate latest model (CTR uplift vs behavior & oracle)
rtb serve           # FastAPI on :8000 (POST /bid, /healthz, /model, /admin/reload)
rtb retrain --once  # one retrain cycle: features → warm-start train → sim gate → promote
```

CI (`.github/workflows/ci.yml`) runs ruff + mypy + pytest + `rtb demo` on Python 3.12.

## Architecture in one paragraph

`data/synth.py` generates a dataset from a `LatentClickModel` (also the simulator's ground
truth, saved to `data/raw/latent.json`). `pipelines/build_features.py` embeds websites/ads
(`embeddings/` — local e5, API, or deterministic hashing fallback), derives user vectors from
engagement, and writes a `FeatureSnapshot` (npz+json in `data/processed/`) — the single
in-memory feature view used by training, simulation, **and** serving. `rl/replay.py` turns won
impressions into one-step transitions; `rl/agent.py` + `rl/networks.py` train
`Q(state, ad_features + learned per-ad id-embedding)`; `registry.py` versions checkpoints under
`registry/` with a `latest.json` pointer. `serving/inference.py` (`BidScorer`) is the shared
scoring core for both the FastAPI app (`serving/app.py`) and offline eval (`sim/evaluate.py`).
`pipelines/retrain_loop.py` re-runs the pipeline on a schedule and flips the registry pointer;
`serving/deps.py` polls the pointer and hot-swaps the model.

## Invariants — do not break these

1. **Feature encoding must be identical in three places.** `features/encode.py` is the source
   of truth; `rl/replay.py` re-implements the same state layout in batched numpy/torch for
   speed, and `sim/env.py` + `serving/inference.py` call `encode` directly. If you change any
   feature (order, normalization, new scalar), update `encode.py` **and** `replay.py` and the
   dim constants (`N_STATE_EXTRA`, `N_AD_EXTRA`) together.
2. **Id-embedding rows are positional.** `QModel.id_emb` row *i* corresponds to
   `FeatureSnapshot.ad_ids[i]` at training time, and `ModelMeta.ad_ids` records that order. Any
   code that pairs a model with a snapshot must ensure the orderings match (this is currently
   *not* validated at load — see ARCHITECTURE.md "Known gaps" before relying on hot-swap).
3. **Embedders must be deterministic and L2-normalized** (`embeddings/base.py`). Affinity and
   site/user match features are plain dot products that assume unit norm.
4. **Everything must keep running offline.** Tests and CI use the hashing embedder + in-memory
   store; never add a hard dependency on network, Redis, Postgres, or model downloads to the
   core path. Heavy deps go in optional extras (`embeddings`, `postgres`) with lazy imports.
5. **Determinism by seed.** Data generation, training, and eval all derive from `cfg.seed`.
   Tests assert reproducibility; keep new randomness behind `np.random.default_rng(seed)` /
   `torch.manual_seed`.

## Gotchas (verified behaviors, not speculation)

- **Config precedence is currently inverted vs. its docs.** `get_config()` passes YAML values
  as init kwargs to a `BaseSettings`, and pydantic-settings gives init kwargs priority over
  env vars. Result: `RTB__SECTION__FIELD` env overrides are **silently ignored for any field
  present in `configs/config.yaml`** (verified empirically). CI's
  `RTB__EMBEDDINGS__PROVIDER=hashing` only works because the local embedder import fails and
  falls back; docker-compose's `RTB__STORE__*` overrides do not take effect at all. If you need
  an override today, edit the YAML or point `RTB_CONFIG_FILE` at an alternate file.
  `POSTGRES_DSN`/`REDIS_URL` are read directly from the environment and do work.
- `get_config()` is `lru_cache`d — call `reset_config_cache()` in tests that tweak env vars.
- The registry root honors `RTB_REGISTRY_DIR` (tests isolate it via an autouse fixture).
- `cfg.sim.candidate_pool` doubles as the **training** CQL candidate-pool size in
  `rl/trainer.py` — changing sim config changes training.
- The Gymnasium env (`sim/env.py`) is exercised only by tests; the training and evaluation
  pipelines do not roll it out. Don't describe it as part of the training loop.
- `Website.base_ctr`/`base_cvr`, smoothed ad CTR, and market stats are computed over the whole
  log window (no temporal split) — fine for the synthetic PoC, but don't cite them as
  leakage-free.
- Windows quirks: prefer `python -m rtb_rl.cli ...` when `make` is unavailable; paths in
  config are repo-relative and resolved via `Config.abspath`.

## Conventions

- Python 3.12, ruff (line length 100, `E,F,I,UP,B,C4,SIM`), mypy on `src` only.
- Pydantic v2 models for all domain schemas (`schemas.py`); dataclasses for internal
  containers (snapshot, dataset, meta).
- Module docstrings carry the design rationale — keep them accurate when changing behavior;
  the README's claims are expected to match the code (this is a portfolio repo; reviewers
  diff prose against code).
- Tests are offline, seed-pinned, and use the tiny `cfg` fixture in `tests/conftest.py`.
  New features need a test in the same style.
