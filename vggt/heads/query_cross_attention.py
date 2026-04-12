"""
QueryCrossAttentionLayer and QueryEvolutionModule for QCOD.

Design (plan D2):
  * Query co-evolution with VGGT backbone: 24 layers of cross-attn → self-attn → FFN
  * Asymmetric dims: queries live in query_dim=512 throughout; image tokens come
    from backbone at token_dim=1024. K/V projections do 1024 → 512.
  * Callback is owned by VGGT (not Aggregator) so freezing "*aggregator*" doesn't
    freeze the query evolution params.
  * Callback signature: callback(queries, tokens_full, patch_start_idx, B, S, P_full, layer_idx)
    → (updated_queries, attn_w_or_None)
  * Special tokens (camera + register) are sliced out INSIDE the callback (plan D2
    "方案 B"). Cross-attention only operates on patch tokens; attn_w is always
    [B, heads, O, S*P_patch].
  * fada_layers controls which layers return attn_w (rest return None so the
    aggregator does not cache them).
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from vggt.layers.layer_scale import LayerScale
from vggt.layers.mlp import Mlp


class QueryCrossAttentionLayer(nn.Module):
    """Single QCOD query evolution block.

    Pre-norm structure:
        queries ← queries + LS1(cross_attn(LN(queries), LN(tokens_patch)))
        queries ← queries + LS2(self_attn(LN(queries)))
        queries ← queries + LS3(ffn(LN(queries)))

    Cross-attention uses asymmetric projection: Q from queries (query_dim), K/V
    from image tokens (token_dim → query_dim). Self-attention and FFN both
    operate in query_dim space.
    """

    def __init__(
        self,
        query_dim: int = 512,
        token_dim: int = 1024,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        init_values: float = 0.01,
    ) -> None:
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.query_dim = query_dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        # --- Cross-attention (queries attend to patch tokens) ---
        self.norm_q_cross = nn.LayerNorm(query_dim)
        self.norm_kv_cross = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(token_dim, query_dim)  # 1024 → 512
        self.v_proj = nn.Linear(token_dim, query_dim)  # 1024 → 512
        self.cross_out = nn.Linear(query_dim, query_dim)
        self.ls_cross = LayerScale(query_dim, init_values=init_values)

        # --- Self-attention among queries ---
        self.norm_self = nn.LayerNorm(query_dim)
        self.self_attn = nn.MultiheadAttention(
            query_dim, num_heads, batch_first=True, bias=True
        )
        self.ls_self = LayerScale(query_dim, init_values=init_values)

        # --- FFN ---
        self.norm_ffn = nn.LayerNorm(query_dim)
        hidden = int(query_dim * mlp_ratio)
        self.ffn = Mlp(query_dim, hidden_features=hidden)
        self.ls_ffn = LayerScale(query_dim, init_values=init_values)

    def forward(
        self,
        queries: torch.Tensor,        # [B, O, query_dim]
        tokens_patch: torch.Tensor,   # [B, S*P_patch, token_dim]
        query_pos: torch.Tensor = None,  # [B, O, query_dim] persistent positional embedding
    ):
        """
        Args:
            query_pos: per-query positional embedding, added to Q and K in
                self-attention to prevent query collapse. This is the initial
                object_query embedding, broadcast-expanded to batch size.
                When provided, self-attn computes:
                    Q = proj(norm(queries) + query_pos)
                    K = proj(norm(queries) + query_pos)
                    V = proj(norm(queries))  ← NO pos bias on values
                This is the standard DETR recipe: pos biases guide WHICH queries
                attend to each other, but don't contaminate WHAT information flows.

        Returns:
            updated_queries: [B, O, query_dim]
            attn_weights:    [B, num_heads, O, S*P_patch]
        """
        B, O, _ = queries.shape
        _, N, _ = tokens_patch.shape

        # ── Cross-attention (manual to get attention weights) ──
        q_norm = self.norm_q_cross(queries)
        kv_norm = self.norm_kv_cross(tokens_patch)

        q = self.q_proj(q_norm).view(B, O, self.num_heads, self.head_dim).transpose(1, 2)
        # q: [B, H, O, D_head]
        k = self.k_proj(kv_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # k, v: [B, H, N, D_head]

        # Scaled dot-product
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # [B, H, O, N]
        attn_weights = F.softmax(attn_logits, dim=-1)
        context = torch.matmul(attn_weights, v)  # [B, H, O, D_head]
        context = context.transpose(1, 2).contiguous().view(B, O, self.query_dim)
        context = self.cross_out(context)

        queries = queries + self.ls_cross(context)

        # ── Self-attention among queries (with positional bias to prevent collapse) ──
        s_norm = self.norm_self(queries)
        if query_pos is not None:
            # DETR-style: add pos to Q and K but NOT V.
            # This biases "who attends to whom" while preserving content purity.
            sa_q = s_norm + query_pos
            sa_k = s_norm + query_pos
        else:
            sa_q = s_norm
            sa_k = s_norm
        sa_out, _ = self.self_attn(sa_q, sa_k, s_norm, need_weights=False)
        queries = queries + self.ls_self(sa_out)

        # ── FFN ──
        ff_out = self.ffn(self.norm_ffn(queries))
        queries = queries + self.ls_ffn(ff_out)

        return queries, attn_weights


class QueryEvolutionModule(nn.Module):
    """Container for the 24 query cross-attention layers + learnable object queries.

    This module is **owned by VGGT**, not by the Aggregator. The Aggregator receives
    a callback via `get_callback()` and calls it at each of its 24 alternating
    attention blocks. Keeping the params outside the aggregator lets us freeze
    ``*aggregator*`` without freezing the query evolution trainables.

    Args:
        num_layers: number of query cross-attention layers (must match aggregator depth).
        query_dim: internal query dimension (D0: 512).
        token_dim: image token dimension from backbone global attention (D0: 1024).
        num_heads: attention heads (D0: 8 → head_dim=64).
        num_queries: number of learnable object queries (D0: 64).
        mlp_ratio: FFN expansion ratio.
        init_values: LayerScale init (matching VGGT convention 0.01).
        fada_layers: Which layers return cross-attn weights for FADA supervision.
            - "none":    all layers return None. Stage A default (zero extra memory).
            - "last":    only the final layer returns attn. Stage C recommended.
            - "all":     all 24 layers return attn. Experimental only (high memory).
            - List[int]: explicit layer indices.
    """

    def __init__(
        self,
        num_layers: int = 24,
        query_dim: int = 512,
        token_dim: int = 1024,
        num_heads: int = 8,
        num_queries: int = 64,
        mlp_ratio: float = 4.0,
        init_values: float = 0.01,
        fada_layers: Union[str, List[int]] = "none",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.query_dim = query_dim
        self.num_queries = num_queries
        self.fada_layers = fada_layers

        # Learnable object query embeddings
        self.object_queries = nn.Embedding(num_queries, query_dim)
        nn.init.xavier_uniform_(self.object_queries.weight)

        # 24 query cross-attention layers
        self.layers = nn.ModuleList(
            [
                QueryCrossAttentionLayer(
                    query_dim=query_dim,
                    token_dim=token_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(num_layers)
            ]
        )

        # Flag used by the callback to decide between checkpoint and direct call.
        # VGGT sets this via self.query_evolution.train() / .eval() propagation.
        self._use_checkpoint = True

    def _should_keep_attn(self, layer_idx: int) -> bool:
        """Decide whether to return attn weights at the given layer.

        Controls memory: returning None lets the aggregator drop attn_w entirely
        (no autograd graph edges), which is the Stage A default (see plan D4).
        """
        if self.fada_layers == "none":
            return False
        if self.fada_layers == "last":
            return layer_idx == self.num_layers - 1
        if self.fada_layers == "all":
            return True
        if isinstance(self.fada_layers, (list, tuple)):
            return layer_idx in self.fada_layers
        return False

    def initial_queries(self, batch_size: int, device=None) -> torch.Tensor:
        """Build the initial query tensor [B, O, query_dim] expanded across the batch."""
        if device is None:
            device = self.object_queries.weight.device
        q = self.object_queries.weight.to(device).unsqueeze(0).expand(batch_size, -1, -1)
        return q.contiguous()

    def get_callback(self):
        """Return the closure that the aggregator calls at each layer.

        Signature (plan D2):
            callback(queries, tokens_full, patch_start_idx, B, S, P_full, layer_idx)
            → (updated_queries, attn_w_or_None)

        The callback:
          1. Slices out special tokens (camera + register) from tokens_full using
             patch_start_idx — cross-attention operates on patch tokens only.
          2. Passes the **initial object_query embedding** as query_pos to every
             layer's self-attention. This persistent positional bias prevents query
             collapse (the classic DETR trick).
          3. Runs the layer (optionally under gradient checkpointing when training).
          4. Returns attn_w only if self._should_keep_attn(layer_idx) is True;
             otherwise returns None so the aggregator drops it.
        """
        # Pre-compute the positional bias once (detached from the query evolution
        # gradient flow — it's a FIXED identity signal, not something to optimize
        # during each forward pass).
        # Shape: [1, O, query_dim], expanded to batch in each call.
        query_pos_base = self.object_queries.weight.unsqueeze(0)  # [1, O, D]

        def callback(
            queries: torch.Tensor,
            tokens_full: torch.Tensor,
            patch_start_idx: int,
            B: int,
            S: int,
            P_full: int,
            layer_idx: int,
        ):
            # Slice out patch tokens.
            P_patch = P_full - patch_start_idx
            C = tokens_full.shape[-1]
            tokens_patch = (
                tokens_full.view(B, S, P_full, C)[:, :, patch_start_idx:, :]
                .reshape(B, S * P_patch, C)
            )

            # Persistent positional bias from initial embeddings (detached so it
            # stays fixed during training — we want a STABLE identity signal,
            # not one that drifts with the optimizer).
            query_pos = query_pos_base.detach().expand(B, -1, -1).to(tokens_full.device)

            layer = self.layers[layer_idx]
            if self.training and self._use_checkpoint:
                queries, attn_w = checkpoint(
                    layer, queries, tokens_patch, query_pos, use_reentrant=False
                )
            else:
                queries, attn_w = layer(queries, tokens_patch, query_pos=query_pos)

            if not self._should_keep_attn(layer_idx):
                return queries, None
            return queries, attn_w

        return callback
