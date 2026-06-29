import numpy as np

from rtb_rl.data import loaders
from rtb_rl.data.synth import generate
from rtb_rl.pipelines.build_features import build_features
from rtb_rl.sim.env import RTBSimEnv


def test_env_reset_and_rollout(cfg):
    loaders.save_dataset(cfg, generate(cfg))
    snap = build_features(cfg)
    latent = loaders.load_latent(cfg)
    env = RTBSimEnv(cfg, snap, latent, horizon=10, pool=6, seed=0)

    obs, _ = env.reset()
    assert obs["state"].ndim == 1
    assert obs["ad_content"].shape[0] == 6
    assert obs["ad_idx"].shape == (6,)

    total = 0.0
    truncated = False
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        assert np.isfinite(reward)
        if truncated:
            break
    assert truncated
