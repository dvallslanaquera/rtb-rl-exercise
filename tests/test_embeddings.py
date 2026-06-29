import numpy as np

from rtb_rl.embeddings import get_embedder
from rtb_rl.embeddings.hashing import HashingEmbedder


def test_hashing_shape_and_determinism():
    e = HashingEmbedder(dim=64)
    a = e.embed(["資産運用と投資の最新情報"])
    b = e.embed(["資産運用と投資の最新情報"])
    assert a.shape == (1, 64)
    assert np.allclose(a, b)
    # L2-normalized
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5


def test_same_topic_is_closer_than_cross_topic():
    # Mirrors Website.content_text() = vertical + title + description. Same-vertical pages share
    # the vertical token and the description template (only the numeric title suffix differs),
    # so they must land closer than a different-vertical page.
    e = HashingEmbedder(dim=256)
    fin1 = "finance 資産運用と投資の最新情報｜finance001 株式・投資信託・NISA・iDeCoなど資産形成を解説するメディア"
    fin2 = "finance 資産運用と投資の最新情報｜finance007 株式・投資信託・NISA・iDeCoなど資産形成を解説するメディア"
    game = "gaming ゲーム攻略と最新情報｜gaming003 新作ゲームのレビュー・攻略・eスポーツ情報を扱うゲームメディア"
    v = e.embed([fin1, fin2, game])
    finance_sim = float(v[0] @ v[1])
    cross_sim = float(v[0] @ v[2])
    assert finance_sim > cross_sim


def test_factory_returns_hashing(cfg):
    e = get_embedder(cfg)
    assert e.name == "hashing"
    assert e.dim == 64
