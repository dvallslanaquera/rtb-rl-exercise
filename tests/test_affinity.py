import numpy as np

from rtb_rl.data import loaders
from rtb_rl.data.synth import generate
from rtb_rl.embeddings import get_embedder
from rtb_rl.features.affinity import compute_topk_affinity
from rtb_rl.features.store import FeatureSnapshot
from rtb_rl.features.user import build_user_features
from rtb_rl.features.website import build_website_features
from rtb_rl.pipelines.build_features import build_features


def test_topk_affinity_shape_and_range(cfg):
    ds = generate(cfg)
    e = get_embedder(cfg)
    wf = build_website_features(e, ds.websites)
    uf = build_user_features(ds.users, wf["ids"], wf["emb"])
    k = cfg.features.affinity_top_k
    df = compute_topk_affinity(uf["ids"], uf["emb"], wf["ids"], wf["emb"], k)
    assert df.height == len(uf["ids"]) * min(k, len(wf["ids"]))
    assert df["affinity"].min() >= -1.01
    assert df["affinity"].max() <= 1.01


def test_snapshot_roundtrip_and_affinity(cfg):
    loaders.save_dataset(cfg, generate(cfg))
    snap = build_features(cfg)
    snap2 = FeatureSnapshot.load(cfg)
    assert snap2.embed_dim == snap.embed_dim
    assert np.allclose(snap.user_emb, snap2.user_emb)
    assert np.allclose(snap.website_emb, snap2.website_emb)
    a = snap2.affinity(snap2.user_ids[0], snap2.website_ids[0])
    assert -1.01 <= a <= 1.01
    # unknown ids -> 0 affinity, no crash
    assert snap2.affinity("nope", snap2.website_ids[0]) == 0.0
