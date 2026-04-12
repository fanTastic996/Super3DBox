"""
QCODHead — top-level orchestrator for the query-evolution 3D detection pipeline.

Pipeline (plan D0 / architecture overview):
  final_tokens [B, S, P_full, 2048]    (from VGGT Aggregator last layer, concat)
  final_queries [B, O, 512]            (from QueryEvolutionModule after 24 layers)
    │
    ├─ slice patch tokens: [B, S, P_patch, 2048] → [B, S*P_patch, 2048]
    │
    ├─→ TokenActivationModule       → activation [B, O, S*P_patch]
    │                                 activation_logits [B, O, S*P_patch] (for BCE)
    ├─→ InstanceGatherer            → instance_repr [B, O, 256]
    ├─→ Direct3DBoxHead             → {center, size, R, rot6d, logits, corners}
    │
    └─→ Return dict with keys: corners, center, size, R, rot6d, logits,
        activation, activation_logits
        (cross_attn_weights is passed through for the loss function — QCODHead
        itself does not modify or consume it, but carries it so compute_qcod_loss
        sees a uniform predictions dict.)

Key design:
  * Only takes ``final_tokens = aggregated_tokens_list[-1]``, NOT the 24-layer list
    (plan D2 rationale: QueryEvolution has already absorbed multi-layer features;
    downstream only needs the richest last-layer concat tokens).
  * Dynamic H_patch / W_patch derived from ``P_patch = P_full - patch_start_idx``.
  * NOTE: H_patch * W_patch must equal P_patch. We reconstruct H_patch and W_patch
    externally via ``images.shape`` in the caller (VGGT), passed in for the
    token_activation layout [B, O, S*P_patch] → [B, O, S*H_patch*W_patch].
    activation itself is stored flat (B, O, S*P_patch) so the loss function can
    reshape it without QCODHead needing H/W.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from vggt.heads.instance_activation import TokenActivationModule, InstanceGatherer
from vggt.heads.box3d_head import Direct3DBoxHead


class QCODHead(nn.Module):
    """Top-level QCOD head: activation → gather → 3D box.

    Args:
        query_dim: dimension of final queries from QueryEvolutionModule (D0: 512).
        token_dim: dimension of concat image tokens from Aggregator last layer
            (D0: 2048 = 2 × backbone_dim = 2 × 1024).
        proj_dim: shared projection dimension for TokenActivation (D0: 256).
        gather_dim: output dimension of InstanceGatherer and input dimension of
            Direct3DBoxHead (D0: 256, = box_in_dim).
        top_k: top-K activated tokens for the attention-pool branch in InstanceGatherer.
    """

    def __init__(
        self,
        query_dim: int = 512,
        token_dim: int = 2048,
        proj_dim: int = 256,
        gather_dim: int = 256,
        top_k: int = 128,
        num_classes: int = 2,
        num_queries: int = 64,
    ) -> None:
        super().__init__()
        assert proj_dim == 256, "plan D0 fixes proj_dim = 256"
        assert gather_dim == 256, "plan D0 fixes gather_dim = box_in_dim = 256"

        self.query_dim = query_dim
        self.token_dim = token_dim
        self.proj_dim = proj_dim
        self.gather_dim = gather_dim

        self.token_activation = TokenActivationModule(
            query_dim=query_dim, token_dim=token_dim, proj_dim=proj_dim
        )
        self.instance_gatherer = InstanceGatherer(
            token_dim=token_dim, proj_dim=gather_dim, top_k=top_k
        )
        self.box_head = Direct3DBoxHead(
            in_dim=gather_dim, hidden_dim=512, num_classes=num_classes,
            num_queries=num_queries,
        )

    def forward(
        self,
        final_tokens: torch.Tensor,                    # [B, S, P_full, 2048]
        patch_start_idx: int,
        final_queries: torch.Tensor,                   # [B, O, query_dim]
        cross_attn_weights: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            final_tokens: ``aggregated_tokens_list[-1]`` — [B, S, P_full, 2048].
                P_full includes special tokens (camera + register) and patch tokens.
            patch_start_idx: int, the index where patch tokens begin in P_full
                (= 1 + num_register_tokens, typically 5).
            final_queries: [B, O, query_dim] — output of QueryEvolutionModule.
            cross_attn_weights: list of [B, heads, O, S*P_patch] tensors saved by the
                callback, or None / empty list when fada_layers="none".
                QCODHead passes this through unchanged for the loss function to consume.

        Returns:
            Dict with keys:
                'corners':             [B, O, 8, 3]
                'center':              [B, O, 3]
                'size':                [B, O, 3]       half-extent
                'R':                   [B, O, 3, 3]
                'rot6d':               [B, O, 6]
                'logits':              [B, O, num_classes]   dim 0=bg, dim 1=fg
                'activation':          [B, O, S*P_patch]     in [0, 1]
                'activation_logits':   [B, O, S*P_patch]     raw logits (for BCE)
                'cross_attn_weights':  the list passed in (possibly empty)
        """
        B, S, P_full, C = final_tokens.shape

        # ── Slice patch tokens (drop camera + register) ──
        # [B, S, P_full, C] → [B, S, P_patch, C] → flatten to [B, S*P_patch, C]
        patch_tokens = final_tokens[:, :, patch_start_idx:, :]          # [B, S, P_patch, C]
        P_patch = patch_tokens.shape[2]
        tokens_flat = patch_tokens.reshape(B, S * P_patch, C)           # [B, S*P_patch, C]

        # ── Token Activation ──
        activation, activation_logits = self.token_activation(
            final_queries, tokens_flat
        )  # [B, O, S*P_patch] each

        # ── Instance Gatherer ──
        instance_repr = self.instance_gatherer(
            activation, tokens_flat
        )  # [B, O, gather_dim]

        # ── Direct 3D Box Head ──
        box_preds = self.box_head(instance_repr)

        out = {
            "corners": box_preds["corners"],
            "center": box_preds["center"],
            "size": box_preds["size"],
            "R": box_preds["R"],
            "rot6d": box_preds["rot6d"],
            "logits": box_preds["logits"],
            "activation": activation,
            "activation_logits": activation_logits,
        }
        if cross_attn_weights is not None:
            out["cross_attn_weights"] = cross_attn_weights
        return out
