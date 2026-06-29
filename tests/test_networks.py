import torch

from rtb_rl.rl.networks import QModel


def test_forward_shapes():
    m = QModel(state_dim=20, ad_content_dim=10, n_ads=15, id_dim=4, hidden_dim=32)
    state = torch.randn(5, 20)
    ad_content = torch.randn(5, 8, 10)
    idx = torch.randint(0, 15, (5, 8))
    q = m(state, ad_content, idx)
    assert q.shape == (5, 8)


def test_qvalues_emb_matches_table_lookup():
    m = QModel(20, 10, 15, id_dim=4, hidden_dim=32).eval()
    state = torch.randn(3, 20)
    ad_content = torch.randn(3, 6, 10)
    idx = torch.randint(0, 15, (3, 6))
    with torch.no_grad():
        q_idx = m(state, ad_content, idx)
        q_emb = m.q_values_emb(state, ad_content, m.id_emb(idx))
    assert torch.allclose(q_idx, q_emb, atol=1e-5)


def test_non_dueling_runs():
    m = QModel(12, 6, 10, id_dim=3, hidden_dim=16, dueling=False)
    q = m(torch.randn(4, 12), torch.randn(4, 5, 6), torch.randint(0, 10, (4, 5)))
    assert q.shape == (4, 5)
