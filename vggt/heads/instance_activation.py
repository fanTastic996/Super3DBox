"""
TokenActivationModule and InstanceGatherer for QCOD.

Design (plan D0 + architecture overview):

  Token Activation
    query_proj:  [B, O, 512]    → [B, O, 256]        (query_dim → proj_dim)
    token_proj:  [B, S*P, 2048] → [B, S*P, 256]      (token_dim → proj_dim)
    sigmoid(dot) → activation [B, O, S*P]
    Also returns pre-sigmoid logits for BCE supervision.

  Instance Gatherer
    Dual-branch pooling for robustness:
      Branch 1 (weighted avg): normalized activation weights × projected tokens
                               → captures global context for each instance.
      Branch 2 (top-K attn):   select top-K most activated tokens per query,
                               then attention-pool them with a learnable query.
                               → captures fine-grained discriminative features.
    Sum both branches → instance_repr [B, O, 256] (= gather_dim = box_in_dim).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenActivationModule(nn.Module):
    """Compute per-instance per-token activation via projected dot product + sigmoid.

    Projects queries and image tokens into a shared D0 proj_dim=256 space, then
    computes per-query per-token activation as sigmoid(q · t^T * temperature).

    The sigmoid (not softmax) activation lets each token independently be "on"
    for a given query — appropriate since an instance's tokens are typically a
    small fraction of the total, and two different instances can share tokens
    at their boundaries.
    """

    def __init__(
        self,
        query_dim: int = 512,
        token_dim: int = 2048,
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim

        self.query_norm = nn.LayerNorm(query_dim)
        self.query_proj = nn.Linear(query_dim, proj_dim)

        self.token_norm = nn.LayerNorm(token_dim)
        self.token_proj = nn.Linear(token_dim, proj_dim)

        # Learnable temperature (initialized to standard 1/sqrt(d) scaling)
        self.temperature = nn.Parameter(torch.tensor(proj_dim ** -0.5))

    def forward(
        self,
        queries: torch.Tensor,        # [B, O, query_dim]
        tokens: torch.Tensor,         # [B, N, token_dim]   (N = S * P_patch)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            activation: [B, O, N] in [0, 1] — sigmoid-normalized token relevance.
            logits:     [B, O, N] — pre-sigmoid logits (for BCE loss).
        """
        q = self.query_proj(self.query_norm(queries))   # [B, O, D]
        t = self.token_proj(self.token_norm(tokens))    # [B, N, D]

        # Scaled dot product → logits
        logits = torch.einsum("bod,bnd->bon", q, t) * self.temperature  # [B, O, N]
        activation = torch.sigmoid(logits)
        return activation, logits


class InstanceGatherer(nn.Module):
    """Pool activated tokens into fixed-size instance representations.

    Dual-branch design:
      * Weighted-average branch: uses activation as soft weights over all tokens.
      * Top-K attention branch: picks top-K most activated tokens per query and
        attention-pools them with a learnable pool-query vector.

    The two branches are summed and projected to gather_dim = 256 = box_in_dim.
    """

    def __init__(
        self,
        token_dim: int = 2048,
        proj_dim: int = 256,       # = gather_dim = box_in_dim
        top_k: int = 128,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.top_k = top_k

        # Projection shared by both branches: 2048 → 256
        self.token_proj = nn.Linear(token_dim, proj_dim)
        self.token_norm = nn.LayerNorm(proj_dim)

        # Branch 2: top-K attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            proj_dim, num_heads, batch_first=True, bias=True
        )
        self.pool_norm = nn.LayerNorm(proj_dim)

        # Final fusion: branch1 + branch2 → LayerNorm → proj_dim
        self.out_norm = nn.LayerNorm(proj_dim)

    def forward(
        self,
        activation: torch.Tensor,     # [B, O, N]  (N = S * P_patch)
        tokens: torch.Tensor,         # [B, N, token_dim]
    ) -> torch.Tensor:
        """
        Returns:
            instance_repr: [B, O, proj_dim]
        """
        B, O, N = activation.shape
        D = self.proj_dim

        # Shared projection of tokens into gather_dim space.
        tokens_proj = self.token_proj(tokens)                     # [B, N, D]
        tokens_proj = self.token_norm(tokens_proj)

        # ── Branch 1: soft weighted average ──
        # Normalize activation per query so we get a proper weighted mean.
        weights = activation / activation.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weighted_avg = torch.einsum("bon,bnd->bod", weights, tokens_proj)  # [B, O, D]

        # ── Branch 2: top-K attention pooling ──
        k = min(self.top_k, N)
        _, topk_idx = activation.topk(k, dim=-1)          # [B, O, k]
        # Gather top-K tokens for each query:
        # tokens_proj: [B, N, D] → expand to [B, O, N, D] → gather index [B, O, k] → [B, O, k, D]
        idx_exp = topk_idx.unsqueeze(-1).expand(B, O, k, D)       # [B, O, k, D]
        tokens_exp = tokens_proj.unsqueeze(1).expand(B, O, N, D)   # [B, O, N, D]
        topk_tokens = torch.gather(tokens_exp, dim=2, index=idx_exp)  # [B, O, k, D]

        # Attention-pool each query's K tokens with a shared learnable query vector.
        # Flatten (B, O) → batch dim for the attention layer.
        topk_flat = topk_tokens.reshape(B * O, k, D)              # [B*O, k, D]
        pool_q = self.pool_query.expand(B * O, 1, D)               # [B*O, 1, D]
        pooled, _ = self.pool_attn(pool_q, topk_flat, topk_flat, need_weights=False)
        # pooled: [B*O, 1, D]
        pooled = pooled.view(B, O, D)
        pooled = self.pool_norm(pooled)

        # Combine both branches.
        instance_repr = self.out_norm(weighted_avg + pooled)       # [B, O, D]
        return instance_repr
