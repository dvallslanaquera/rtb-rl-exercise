"""Dueling Double-DQN agent with a Conservative Q-Learning (CQL) penalty.

Loss = TD error + ``cql_alpha`` · CQL(H) penalty, where::

    CQL(H) = mean_i [ logsumexp_a Q(s_i, a)  -  Q(s_i, a_logged_i) ]

pushes Q *down* for actions not taken in the logs while keeping the behavior action's value
up — the standard offline-RL correction against value overestimation on out-of-distribution
actions. The TD target is one-step (terminal) for logged impressions, and uses the Double-DQN
bootstrap (online net selects, target net evaluates) when next-state transitions are supplied
(the simulator's budget-paced episodes during retraining).
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from rtb_rl.config import RLConfig
from rtb_rl.rl.networks import QModel


class DQNAgent:
    def __init__(
        self, state_dim: int, ad_content_dim: int, n_ads: int, rl: RLConfig,
        id_dim: int = 16, device: str = "cpu",
    ) -> None:
        self.rl = rl
        self.device = torch.device(device)
        self.online = QModel(
            state_dim, ad_content_dim, n_ads, id_dim, rl.hidden_dim, rl.dueling
        ).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=rl.lr)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        state = batch["state"].to(self.device)
        ad_content = batch["ad_content"].to(self.device)
        pool_idx = batch["pool_idx"].to(self.device)
        reward = batch["reward"].to(self.device)

        q_all = self.online(state, ad_content, pool_idx)  # (B, C)
        q_logged = q_all[:, 0]  # column 0 == logged action

        if "next_state" in batch and self.rl.gamma > 0:
            y = self._bootstrap_target(batch, reward)
        else:
            y = reward  # one-step / terminal

        td_loss = nn.functional.mse_loss(q_logged, y)
        # CQL(H): conservative penalty over the candidate action set.
        cql_loss = (torch.logsumexp(q_all, dim=1) - q_logged).mean()
        loss = td_loss + self.rl.cql_alpha * cql_loss
        info = {
            "td_loss": float(td_loss.detach()),
            "cql_loss": float(cql_loss.detach()),
            "q_mean": float(q_all.mean().detach()),
        }
        return loss, info

    @torch.no_grad()
    def _bootstrap_target(
        self, batch: dict[str, torch.Tensor], reward: torch.Tensor
    ) -> torch.Tensor:
        ns = batch["next_state"].to(self.device)
        nac = batch["next_ad_content"].to(self.device)
        npool = batch["next_pool_idx"].to(self.device)
        done = batch.get("done")
        done_t = (done.to(self.device) if done is not None else torch.zeros_like(reward))
        # Double DQN: select with the online net, evaluate with the target net.
        a_star = self.online(ns, nac, npool).argmax(dim=1, keepdim=True)
        q_next = self.target(ns, nac, npool).gather(1, a_star).squeeze(1)
        return reward + self.rl.gamma * (1.0 - done_t) * q_next

    def update(self, batch: dict[str, torch.Tensor]) -> dict:
        self.online.train()
        loss, info = self.compute_loss(batch)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self._soft_update()
        info["loss"] = float(loss.detach())
        return info

    def _soft_update(self) -> None:
        tau = self.rl.target_update_tau
        for tp, op in zip(self.target.parameters(), self.online.parameters(), strict=True):
            tp.data.mul_(1 - tau).add_(tau * op.data)

    def id_table(self) -> torch.Tensor:
        return self.online.id_emb.weight.detach().cpu()
