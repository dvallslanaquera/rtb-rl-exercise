import numpy as np
import torch

from rtb_rl.config import RLConfig
from rtb_rl.rl.agent import DQNAgent


def _fixed_batch(b=64, c=6, ds=16, dac=8, n_ads=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "state": torch.randn(b, ds, generator=g),
        "ad_content": torch.randn(b, c, dac, generator=g),
        "pool_idx": torch.randint(0, n_ads, (b, c), generator=g),
        "reward": torch.rand(b, generator=g),
    }


def test_update_reduces_loss_on_fixed_batch():
    rl = RLConfig(lr=1e-3, cql_alpha=0.5)
    agent = DQNAgent(16, 8, 12, rl, id_dim=4)
    batch = _fixed_batch()
    first, _ = agent.compute_loss(batch)
    for _ in range(60):
        agent.update(batch)
    last, info = agent.compute_loss(batch)
    assert float(last.detach()) < float(first.detach())
    assert np.isfinite(info["cql_loss"])


def test_bootstrap_target_path_runs():
    rl = RLConfig(gamma=0.9)
    agent = DQNAgent(16, 8, 12, rl, id_dim=4)
    batch = _fixed_batch()
    nxt = _fixed_batch(seed=1)
    batch.update(
        {
            "next_state": nxt["state"],
            "next_ad_content": nxt["ad_content"],
            "next_pool_idx": nxt["pool_idx"],
            "done": torch.zeros(64),
        }
    )
    loss, info = agent.compute_loss(batch)
    assert np.isfinite(float(loss.detach()))
