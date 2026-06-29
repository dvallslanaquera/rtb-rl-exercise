"""Dueling Q-network operating on (state, ad-features) pairs.

Scoring ``Q(state, ad_features)`` and argmaxing over a candidate set — rather than a fixed
per-ad output head — is what makes a variable/growing ad inventory and cold-start tractable.

A learned per-ad **id-embedding** is concatenated to the content-based ad features. It absorbs
the residual ad appeal that content alone can't explain, and is the quantity cold-start borrows
from similar ads (see :mod:`rtb_rl.rl.cold_start`). Known ads index the embedding table; new
ads are scored by passing a neighbor-averaged id-embedding vector directly.
"""

from __future__ import annotations

import torch
from torch import nn


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, out_dim),
    )


class QModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        ad_content_dim: int,
        n_ads: int,
        id_dim: int = 16,
        hidden_dim: int = 256,
        dueling: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.ad_content_dim = ad_content_dim
        self.id_dim = id_dim
        self.dueling = dueling

        self.id_emb = nn.Embedding(n_ads, id_dim)
        nn.init.normal_(self.id_emb.weight, std=0.1)

        self.state_trunk = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        ad_full_dim = ad_content_dim + id_dim
        self.adv_mlp = _mlp(hidden_dim + ad_full_dim, hidden_dim, 1)
        if dueling:
            self.value_mlp = _mlp(hidden_dim, hidden_dim, 1)

    # ---- core ----
    def _q_from_idemb(
        self, state: torch.Tensor, ad_content: torch.Tensor, id_emb: torch.Tensor
    ) -> torch.Tensor:
        """state (B, Ds); ad_content (B, C, Dac); id_emb (B, C, id_dim) -> Q (B, C)."""
        b, c, _ = ad_content.shape
        h_s = self.state_trunk(state)  # (B, H)
        ad_full = torch.cat([ad_content, id_emb], dim=-1)  # (B, C, Dac+id)
        h_exp = h_s.unsqueeze(1).expand(-1, c, -1)  # (B, C, H)
        adv = self.adv_mlp(torch.cat([h_exp, ad_full], dim=-1)).squeeze(-1)  # (B, C)
        if not self.dueling:
            return adv
        value = self.value_mlp(h_s).squeeze(-1).unsqueeze(1)  # (B, 1)
        return value + adv - adv.mean(dim=1, keepdim=True)

    def forward(
        self, state: torch.Tensor, ad_content: torch.Tensor, ad_idx: torch.Tensor
    ) -> torch.Tensor:
        """Score candidates by their id-embedding-table index. ad_idx (B, C) long -> Q (B, C)."""
        return self._q_from_idemb(state, ad_content, self.id_emb(ad_idx))

    def q_values_emb(
        self, state: torch.Tensor, ad_content: torch.Tensor, id_emb: torch.Tensor
    ) -> torch.Tensor:
        """Score candidates by explicit id-embedding vectors (cold start / serving)."""
        return self._q_from_idemb(state, ad_content, id_emb)
