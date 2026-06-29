import rtb_rl.serving.deps as deps_mod
from rtb_rl.data import loaders
from rtb_rl.data.synth import generate
from rtb_rl.embeddings import get_embedder
from rtb_rl.features.store import FeatureSnapshot
from rtb_rl.pipelines.build_features import build_features
from rtb_rl.pipelines.train import train
from rtb_rl.registry import ModelRegistry
from rtb_rl.serving.inference import BidScorer, ColdAd


def _bootstrap(cfg):
    loaders.save_dataset(cfg, generate(cfg))
    build_features(cfg)
    train(cfg)


def test_bid_endpoint(cfg):
    from fastapi.testclient import TestClient

    from rtb_rl.serving.app import app
    from rtb_rl.serving.deps import ServingState

    _bootstrap(cfg)
    state = ServingState(cfg)
    state.load()
    deps_mod._state = state
    try:
        with TestClient(app) as client:
            health = client.get("/healthz").json()
            assert health["ready"] is True
            assert health["model_version"] == state.version

            resp = client.post(
                "/bid",
                json={
                    "request_id": "r1",
                    "website_id": "w0000",
                    "placement": "header",
                    "user_id": "u000000",
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["ad_id"] in state.snapshot.ad_ids
            assert body["bid_price_jpy"] >= 0.0
            assert body["latency_ms"] >= 0.0

            # unknown website -> 404
            bad = client.post(
                "/bid",
                json={
                    "request_id": "r2",
                    "website_id": "nonexistent",
                    "placement": "header",
                    "user_id": "u000000",
                },
            )
            assert bad.status_code == 404
    finally:
        deps_mod._state = None


def test_coldstart_scoring(cfg):
    _bootstrap(cfg)
    snap = FeatureSnapshot.load(cfg)
    model, meta = ModelRegistry(cfg).load_latest()
    scorer = BidScorer(snap, model, meta, cfg)

    # held-out cold-start ad is scorable via neighbor borrowing
    cold_ad = next(a for a in loaders.load_ads(cfg) if a.is_coldstart)
    emb = get_embedder(cfg).embed([cold_ad.content_text()])[0]
    res = scorer.score(
        snap.website_ids[0],
        snap.user_ids[0],
        "header",
        candidate_ad_ids=snap.ad_ids[:5],
        cold_candidates=[ColdAd(cold_ad.ad_id, emb, cold_ad.bid_cap_jpy)],
    )
    assert cold_ad.ad_id in res.q_by_ad

    # brand-new user falls back to the content-aligned cold prior
    res2 = scorer.score(snap.website_ids[0], "u999999", "header",
                        candidate_ad_ids=snap.ad_ids[:5])
    assert res2.cold_start is True
