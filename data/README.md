# data/

Generated artifacts (git-ignored):

- `raw/` — synthetic dataset: `websites.parquet`, `users.parquet`, `ads.parquet`,
  `bid_logs.parquet`, and `latent.json` (the ground-truth click model shared with the simulator).
- `processed/` — `feature_snapshot.npz` + `.json` (the serving/training feature view),
  `affinity.parquet` (offline top-K user↔website affinity), and `features.db` (sqlite store
  when `store.feature_backend=sqlite`).

Regenerate with `rtb generate-data && rtb build-features` (or just `rtb demo`).
