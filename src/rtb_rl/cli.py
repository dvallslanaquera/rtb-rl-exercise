"""Command-line entrypoint (``rtb``) wiring the full pipeline.

    rtb generate-data     # synthesize websites/users/ads/bid-logs
    rtb build-features     # embeddings + affinity + market context -> snapshot
    rtb train              # offline Dueling Double-DQN + CQL -> registry
    rtb sim                # evaluate the learned policy (CTR uplift) in the simulator
    rtb serve              # FastAPI bidding service
    rtb retrain [--once]   # every-N-hours retrain + sim-gated promotion
    rtb demo               # run the whole offline pipeline end-to-end and print results
"""

from __future__ import annotations

import logging

import typer

from rtb_rl.config import get_config

app = typer.Typer(add_completion=False, help="Deep-RL RTB optimization engine (PoC)")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


@app.command("generate-data")
def generate_data() -> None:
    """Generate and persist the synthetic dataset."""
    _setup_logging()
    from rtb_rl.data import loaders
    from rtb_rl.data.synth import generate

    cfg = get_config()
    cfg.ensure_dirs()
    ds = generate(cfg)
    loaders.save_dataset(cfg, ds)
    won = sum(b.won for b in ds.bid_logs)
    clicks = sum(b.click for b in ds.bid_logs)
    typer.echo(
        f"Generated: {len(ds.websites)} sites, {len(ds.users)} users, {len(ds.ads)} ads "
        f"({sum(a.is_coldstart for a in ds.ads)} cold-start), {len(ds.bid_logs)} bid logs "
        f"(won={won}, clicks={clicks})."
    )


@app.command("build-features")
def build_features_cmd() -> None:
    """Compute embeddings, affinity and the feature snapshot."""
    _setup_logging()
    from rtb_rl.pipelines.build_features import build_features

    snap = build_features(get_config())
    typer.echo(
        f"Features built (embed_dim={snap.embed_dim}): "
        f"{len(snap.website_ids)} sites, {len(snap.user_ids)} users, {len(snap.ad_ids)} ads."
    )


@app.command("train")
def train_cmd(
    warm_start: str = typer.Option(None, help="Registry version to warm-start from"),
) -> None:
    """Train the offline DQN and register the model."""
    _setup_logging()
    from rtb_rl.pipelines.train import train

    meta = train(get_config(), warm_start_version=warm_start)
    if meta:
        typer.echo(f"Registered model {meta.version}: {meta.metrics}")


@app.command("sim")
def sim_cmd(n: int = typer.Option(None, help="Number of eval requests")) -> None:
    """Evaluate the latest model in the offline simulator."""
    _setup_logging()
    from rtb_rl.data import loaders
    from rtb_rl.features.store import FeatureSnapshot
    from rtb_rl.registry import ModelRegistry
    from rtb_rl.serving.inference import BidScorer
    from rtb_rl.sim import evaluate as sim_eval

    cfg = get_config()
    snap = FeatureSnapshot.load(cfg)
    model, meta = ModelRegistry(cfg).load_latest()
    scorer = BidScorer(snap, model, meta, cfg)
    result = sim_eval.evaluate(cfg, snap, scorer, loaders.load_latent(cfg), n_requests=n)
    _print_eval(result)


@app.command("serve")
def serve_cmd() -> None:
    """Run the FastAPI bidding service."""
    _setup_logging()
    import uvicorn

    cfg = get_config()
    uvicorn.run("rtb_rl.serving.app:app", host=cfg.serving.host, port=cfg.serving.port)


@app.command("retrain")
def retrain_cmd(
    once: bool = typer.Option(False, "--once", help="Run a single cycle and exit"),
) -> None:
    """Run the every-N-hours retraining loop (or a single cycle)."""
    _setup_logging()
    from rtb_rl.pipelines import retrain_loop

    cfg = get_config()
    if once:
        result = retrain_loop.run_once(cfg)
        typer.echo(f"Retrain cycle: {result}")
    else:
        retrain_loop.run_loop(cfg)


@app.command("demo")
def demo() -> None:
    """Run the entire offline pipeline end-to-end and print the headline results."""
    _setup_logging()
    from rtb_rl.data import loaders
    from rtb_rl.data.synth import generate
    from rtb_rl.features.store import FeatureSnapshot
    from rtb_rl.pipelines.build_features import build_features
    from rtb_rl.pipelines.train import train
    from rtb_rl.registry import ModelRegistry
    from rtb_rl.serving.inference import BidScorer
    from rtb_rl.sim import evaluate as sim_eval

    cfg = get_config()
    cfg.ensure_dirs()

    typer.echo("\n[1/4] Generating synthetic data ...")
    loaders.save_dataset(cfg, generate(cfg))

    typer.echo("[2/4] Building features (embeddings + affinity) ...")
    build_features(cfg)

    typer.echo("[3/4] Training Dueling Double-DQN + CQL ...")
    train(cfg)

    typer.echo("[4/4] Evaluating in the offline simulator ...\n")
    snap = FeatureSnapshot.load(cfg)
    model, meta = ModelRegistry(cfg).load_latest()
    scorer = BidScorer(snap, model, meta, cfg)
    result = sim_eval.evaluate(cfg, snap, scorer, loaders.load_latent(cfg))
    _print_eval(result)

    cold = sim_eval.coldstart_demo(cfg, snap, scorer)
    if cold:
        typer.echo(
            f"\nCold-start: held-out ad {cold['cold_ad_id']} ({cold['category']}) "
            f"ranked #{cold['rank_among_candidates']}/{cold['n_candidates']} "
            f"(Q={cold['cold_ad_q']:.4f}) on a matching site -- scored from neighbors, "
            f"no history."
        )


def _print_eval(result: dict) -> None:
    typer.echo("=" * 60)
    typer.echo(f"  Eval requests        : {result['n_requests']}")
    typer.echo(f"  Behavior policy CTR  : {result['behavior_ctr'] * 100:.3f}%  (random)")
    typer.echo(f"  Learned policy CTR   : {result['learned_ctr'] * 100:.3f}%  (DQN argmax)")
    typer.echo(f"  Oracle ceiling CTR   : {result['oracle_ctr'] * 100:.3f}%  (best-in-pool)")
    typer.echo(f"  >> CTR uplift        : {result['uplift_pct']:+.1f}% vs behavior")
    typer.echo(f"  >> Oracle gap closed : {result['oracle_gap_closed_pct']:.1f}%")
    if result.get("snips_logged_uplift_pct") is not None:
        typer.echo(f"  SNIPS logged uplift  : {result['snips_logged_uplift_pct']:+.1f}%")
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
