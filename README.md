# rtb-rl — Deep-RL low-latency RTB optimization engine (PoC)

Because RL doesn't need to be the unpopular guy at the classroom.

A reinforcement-learning engine for **real-time bidding (RTB)** ad-yield optimization. For each
bid request it selects the advertisement with the **highest probability of a click** — using
website context, user engagement, and historical auction logs — serves the decision under a
**~10 ms SLA**, and **retrains every *N* hours** to track market drift (new competitor
campaigns, budget/pacing changes). It also handles the **cold-start** problem for brand-new
ads, campaigns and users.

This is a self-contained rebuild driven entirely by **synthetic data**, so the whole pipeline
runs locally end-to-end with no proprietary data and no external services.

> Stack (all 2024-era): Python 3.12 · PyTorch 2 · FastAPI · Gymnasium · LangChain +
> multilingual-e5 · Redis · PostgreSQL · Vertex AI · Docker · Terraform · GitHub Actions.

---

## Results (from `rtb demo`, hashing embedder, seed 42)

```
  Behavior policy CTR  : 13.301%  (random selection — the logging policy)
  Learned policy CTR   : 40.022%  (Dueling Double-DQN + CQL, argmax-Q)
  Oracle ceiling CTR   : 52.489%  (best ad in pool under the true click model)
  >> CTR uplift        : +200.9% vs behavior
  >> Oracle gap closed : 68.2%
  SNIPS logged uplift  : +118.3%

Cold-start: a held-out ad with NO history ranks #2/41 on a matching site,
            scored purely from its content neighbors.

Serving latency (in-process, CPU): p50 0.77 ms · p95 1.05 ms · p99 1.14 ms
```

---

## How it works

```
                          OFFLINE (batch, every N hours)                         ONLINE (10ms)
  ┌───────────────┐   ┌───────────────────────────┐   ┌──────────────────┐   ┌──────────────┐
  │ synthetic     │   │ embeddings (LangChain/e5,  │   │ Dueling Double-DQN│  │ FastAPI /bid │
  │ data:         │──▶│ hashing fallback)          │──▶│ + CQL  (offline   │─▶│ argmax over  │
  │ sites/users/  │   │ → user↔site AFFINITY (cos) │   │ batch RL on WON   │  │ candidate ads│
  │ ads/bid-logs  │   │ → market context           │   │ impressions)      │  │ warm model + │
  └───────────────┘   │ → FeatureSnapshot          │   └─────────┬─────────┘  │ Redis cache  │
         │            └───────────────────────────┘             │            └──────┬───────┘
         │                                                       ▼                   │ hot-swap
         │            ┌───────────────────────────┐   ┌──────────────────┐          │
         └───────────▶│ Gymnasium sim (auction +   │◀──│ model registry   │◀─────────┘
            ground-   │ click + budget pacing)     │   │ (versioned)      │   sim-gated promote
            truth     │ → CTR-uplift eval / gate   │   └──────────────────┘
            click     └───────────────────────────┘
            model
```

### Key design decisions

- **DQN over *ad features*, not a fixed action head.** The Q-network scores
  `Q(state, ad_features)` and the server argmaxes over the eligible candidate ads. Representing
  an ad by its features/embeddings (rather than a per-ad output unit) is what makes a
  variable/growing inventory and cold-start tractable. Architecture: **Dueling** value/advantage
  heads, **Double-DQN** targets, and a **Conservative Q-Learning (CQL)** penalty — the standard
  offline/batch-RL correction so the model doesn't overvalue actions absent from the logs.
- **Click-probability objective.** Training uses **won impressions only** (a click is
  observable only when the ad was shown), so `Q(s,a)` learns expected click value *given the ad
  is served* — exactly the ranking the selector needs. Bidding is a separate value-based step.
- **Why RL and not a plain bandit.** Campaign **budget/pacing** is part of the state, so each
  served impression depletes budget and couples successive requests into an episode — making
  `gamma > 0` meaningful. The offline log fit is one-step; the Gymnasium simulator provides the
  sequential, budget-paced MDP (and the Double-DQN bootstrap path) for fine-tuning.
- **Embeddings & offline affinity.** Website = embed(vertical + JP title + description); user =
  engagement-weighted mean of engaged-site embeddings. **Affinity = cosine**, computed offline
  (top-K per user) and hot-cached. A hybrid embedder defaults to a local multilingual model
  (multilingual-e5 via LangChain), can switch to a hosted API, and falls back to a deterministic
  **character-n-gram hashing** embedder so everything runs offline / in CI with no download.
- **Cold start.** A learned per-ad **id-embedding** captures residual ad appeal; a brand-new ad
  borrows a similarity-weighted average of its content-neighbors' id-embeddings and CTR prior
  (kNN), so it is scored sensibly on impression #1. New users fall back to a content-aligned
  prior that decays as real engagement accrues.
- **10 ms serving.** Features are read from a warm in-process snapshot (dict lookup), the model
  stays in memory, and scoring is a single batched `torch.inference_mode` matmul over the
  candidates — **no Postgres on the hot path**. A background poller hot-swaps the model when the
  retrain loop promotes a new version.

---

## Quickstart

Requires **Python 3.12** (PyTorch has no 3.13/3.14 wheels yet). No GPU, no network needed for
the demo (it uses the hashing embedder + in-memory store).

```bash
py -3.12 -m venv .venv                 # Windows;  python3.12 -m venv .venv  on *nix
.venv\Scripts\activate                 # source .venv/bin/activate  on *nix

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

rtb demo                               # generate → features → train → evaluate (end-to-end)
```

`rtb demo` prints the CTR-uplift table and a cold-start example shown above.

### Individual stages

```bash
rtb generate-data      # synthesize websites / users / ads / bid logs  → data/raw/
rtb build-features     # embeddings + affinity + market context        → data/processed/
rtb train              # offline Dueling Double-DQN + CQL              → registry/
rtb sim                # evaluate the latest model (CTR uplift)
rtb serve              # FastAPI bidding service on :8000
rtb retrain --once     # one retrain cycle (warm-start + sim-gated promote)
rtb retrain            # continuous loop every retrain.interval_hours (APScheduler)
```

Equivalent `make` targets exist (`make demo`, `make train`, …) and thin `scripts/*.py` wrappers.

### Serving

```bash
rtb serve
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/bid -H 'content-type: application/json' \
  -d '{"request_id":"r1","website_id":"w0000","placement":"header","user_id":"u000001"}'
# → {"ad_id":"ad011","bid_price_jpy":155.2,"predicted_click_value":1.2,"model_version":...,"latency_ms":0.8}
```

---

## Real-infra path (Docker / Postgres / Redis)

`docker compose up --build` brings up Postgres (durable feature store) + Redis (hot cache) +
the API + a retrainer; a one-shot `bootstrap` service seeds data/features/model:

```bash
docker compose up --build
curl -s localhost:8000/healthz
```

`infra/terraform/` (Cloud Run + Memorystore + Cloud SQL + Artifact Registry) and
`infra/vertex/pipeline.py` (Vertex AI Pipelines DAG) document the GCP topology as reviewed
stubs. The every-N-hours retrain DAG maps 1:1 onto a Vertex Pipelines schedule.

---

## Configuration

All knobs live in [configs/config.yaml](configs/config.yaml); every field is overridable by env
var `RTB__SECTION__FIELD` (e.g. `RTB__EMBEDDINGS__PROVIDER=local`,
`RTB__STORE__CACHE_BACKEND=redis`, `RTB__RETRAIN__INTERVAL_HOURS=6`). Secrets (`POSTGRES_DSN`,
`REDIS_URL`, API keys) come from the environment — see [.env.example](.env.example).

Notable: `embeddings.provider` (`local`|`api`|`hashing`), `store.feature_backend`
(`memory`|`sqlite`|`postgres`), `store.cache_backend` (`memory`|`redis`), `rl.cql_alpha`,
`rl.gamma`, `retrain.interval_hours`, `retrain.uplift_gate`.

For real multilingual embeddings: `pip install -e ".[embeddings]"` and set
`RTB__EMBEDDINGS__PROVIDER=local` (downloads multilingual-e5).

---

## Project layout

```
src/rtb_rl/
  config.py schemas.py registry.py cli.py
  data/        synth.py (latent click model) · loaders.py (parquet)
  embeddings/  base · local (e5) · api (vertex/openai) · hashing · factory
  features/    website · user · affinity (offline top-K) · encode · store (snapshot/SQL/Redis)
  rl/          networks (dueling) · replay (offline dataset) · agent (Double-DQN+CQL) ·
               trainer · cold_start (kNN priors)
  sim/         env (Gymnasium, budget-paced) · evaluate (CTR uplift + SNIPS)
  serving/     app (FastAPI) · inference (BidScorer) · deps (hot-swap) · cache
  pipelines/   build_features · train · retrain_loop
tests/         17 tests — fully offline (hashing embedder + in-memory store)
infra/         terraform/ (GCP stub) · vertex/ (pipeline stub)
configs/ scripts/ Dockerfile docker-compose.yml .github/workflows/ci.yml
```

---

## Testing & quality

```bash
pytest -q                       # 17 tests, offline, ~9s
ruff check src tests scripts
mypy src
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs lint + types + tests + an
end-to-end `rtb demo` smoke test on Python 3.12.

---

## Notes & limitations (it's a PoC)

- Synthetic data is generated from a known latent click model that is also the simulator's
  ground truth, so reported uplift is an upper-bound-style sanity signal, not a production claim.
- Offline log training is one-step (independent impressions); the sequential `gamma>0` path is
  exercised via the simulator. SNIPS is a coarse, high-variance off-policy check.
- The hashing embedder is purely lexical; install the `[embeddings]` extra for semantic
  multilingual embeddings (which materially improves cold-start neighbor quality).
