"""
Direct3DBoxHead for QCOD — regresses 3D boxes in the normalized first-frame camera system.

Design (plan D1):
  * Input: instance_repr [B, O, box_in_dim=256] from InstanceGatherer.
  * Outputs per query:
      - center (3)              — xyz in normalized coords (first-frame camera, / avg_scale)
      - log_size (3) → size (3) — half-extent via exp(clamp(·, max=5))
      - rot6d (6)    → R (3,3)  — 6D continuous rotation representation (Gram-Schmidt)
      - class_logits (2)        — dim 0 = background, dim 1 = foreground
  * 8 corners derived from (center, size, R) for loss computation & inference.

Key conventions (plan Phase 2 bugfixes + Phase 3+ overfit debug):
  * class_logits layout: dim 0 = background, dim 1 = foreground (matches compute_qcod_loss).
  * size is HALF-EXTENT (not full extent). Matches gt_scale which the loader divides by 2.
  * center_mlp / class_head use default Kaiming init for symmetry breaking.
    size_mlp and rot_mlp use 1e-2 small-init so initial size ≈ 1 and rot ≈ identity
    (rot also gets a fixed identity_bias added in forward). Centers use full diversity
    so the 64 queries do NOT collapse to the same prediction during early training
    (the classic DETR query symmetry trap).
  * Corner ordering follows CA1M face-walk convention:
        bottom CCW: [0,1,2,3] = (-1,-1,-1)→(+1,-1,-1)→(+1,+1,-1)→(-1,+1,-1)
        top CCW:    [4,5,6,7] = (-1,-1,+1)→(+1,-1,+1)→(+1,+1,+1)→(-1,+1,+1)
        edges: 0-1-2-3-0, 4-5-6-7-4, 0-4, 1-5, 2-6, 3-7
"""

import sys
import os

# Ensure qcod_utils is importable (it lives under training/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRAINING_DIR = os.path.join(_REPO_ROOT, "training")
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

import torch
import torch.nn as nn

from qcod_utils import rotation_6d_to_matrix


class _MLP(nn.Module):
    """Simple MLP: in_dim → hidden → hidden → out_dim."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        assert num_layers >= 2
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _center_size_R_to_corners(
    center: torch.Tensor,   # [B, O, 3]
    size: torch.Tensor,     # [B, O, 3]    half-extent
    R: torch.Tensor,        # [B, O, 3, 3]
) -> torch.Tensor:
    """Derive the 8 corners of an OBB from (center, half-extent, rotation).

    Corner ordering (matches CA1M convention, verified against
    /data1/fanwg/CA1M-dataset/.../instances/186.json — "face-walk" layout):

        Local-frame sign pattern:
          [0] (-1,-1,-1)  [1] (+1,-1,-1)  [2] (+1,+1,-1)  [3] (-1,+1,-1)  ← bottom CCW
          [4] (-1,-1,+1)  [5] (+1,-1,+1)  [6] (+1,+1,+1)  [7] (-1,+1,+1)  ← top CCW

        Edges:
          bottom ring: 0-1-2-3-0
          top ring:    4-5-6-7-4
          verticals:   0-4, 1-5, 2-6, 3-7

        corner = center + (signs * half_extent) @ R.T
    """
    device = center.device
    signs = torch.tensor(
        [
            [-1, -1, -1],   # 0
            [+1, -1, -1],   # 1
            [+1, +1, -1],   # 2
            [-1, +1, -1],   # 3
            [-1, -1, +1],   # 4
            [+1, -1, +1],   # 5
            [+1, +1, +1],   # 6
            [-1, +1, +1],   # 7
        ],
        dtype=center.dtype,
        device=device,
    )  # [8, 3]

    # Scale signs by half-extent → [B, O, 8, 3]
    offsets = signs[None, None, :, :] * size[:, :, None, :]          # [B, O, 8, 3]
    # Rotate: offsets @ R.T → [B, O, 8, 3]
    # We want, for each local offset o (column 3-vector), the world-frame rotated
    # vector R @ o. In einsum: (b,o,8,3) . (b,o,3,3) → (b,o,8,3) where the output[...,i]
    # = sum_j R[...,i,j] * offsets[...,j]. Written as einsum:
    #   rotated[b,o,k,i] = sum_j offsets[b,o,k,j] * R[b,o,i,j]
    rotated = torch.einsum("bokj,boij->boki", offsets, R)            # [B, O, 8, 3]
    corners = rotated + center[:, :, None, :]                         # [B, O, 8, 3]
    return corners


class Direct3DBoxHead(nn.Module):
    """Regress 3D boxes directly in the normalized first-frame camera system.

    Args:
        in_dim: input feature dim (must equal gather_dim / box_in_dim per D0 = 256).
        hidden_dim: MLP hidden dim.
        num_classes: number of class logits (2 for binary fg/bg).
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 512,
        num_classes: int = 2,
        num_queries: int = 64,
    ) -> None:
        super().__init__()
        assert in_dim == 256, (
            "Direct3DBoxHead.in_dim must equal box_in_dim=256 (plan D0). "
            "If you need to change this, also update InstanceGatherer.proj_dim."
        )
        self.in_dim = in_dim
        self.num_queries = num_queries

        self.center_mlp = _MLP(in_dim, hidden_dim, 3, num_layers=3)
        self.size_mlp = _MLP(in_dim, hidden_dim, 3, num_layers=3)
        self.rot_mlp = _MLP(in_dim, hidden_dim, 6, num_layers=3)
        self.class_head = nn.Linear(in_dim, num_classes)

        # Project per-query anchor [3] → in_dim, added to instance_repr to inject
        # per-query identity into the shared MLPs (Phase 3+ overfit fix #4).
        self.anchor_proj = nn.Linear(3, in_dim)

        # ── Per-query spatial anchor (non-trainable, prevents query collapse) ──
        # Each of O queries gets its own 3D anchor point. The MLP predicts a
        # RESIDUAL on top of this anchor:  center = anchor + mlp(instance_repr).
        #
        # CRITICAL: anchor is a BUFFER (non-trainable), NOT a Parameter.
        # An earlier draft made it learnable, and the optimizer + AdamW
        # weight_decay + Hungarian gradient signal collapsed all 64 anchors
        # to the same point within ~500 steps (overfit debug, 3rd iteration).
        # Since the anchor's only job is to maintain per-query spatial identity,
        # making it non-trainable is both correct and safer.
        #
        # Initialized to a regular grid spanning [-1, 1]^3, which covers the
        # typical normalized scene extent. Each query has a fixed spatial
        # "home base"; the MLP residual learns the GT-specific refinement.
        self.register_buffer(
            "query_anchor", self._make_grid_anchors(num_queries), persistent=True
        )

        self._init_last_layers()

    @staticmethod
    def _make_grid_anchors(num_queries: int) -> torch.Tensor:
        """Create a regular 3D grid of anchor points in [-1, 1]^3.

        For O=64: 4×4×4=64 evenly spaced points.
        For other O: use cube-root rounding + pad with random uniform.
        """
        import math
        n_per_axis = round(num_queries ** (1 / 3))
        n_grid = n_per_axis ** 3

        # Build grid
        coords = torch.linspace(-1.0, 1.0, n_per_axis)
        gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
        grid = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # [n_grid, 3]

        if n_grid >= num_queries:
            return grid[:num_queries]
        else:
            # Pad with random uniform
            extra = torch.empty(num_queries - n_grid, 3).uniform_(-1.0, 1.0)
            return torch.cat([grid, extra], dim=0)

    def _init_last_layers(self) -> None:
        """Initialize last layers for stable training.

        With the per-query anchor, center_mlp predicts a RESIDUAL:
            center = query_anchor + center_mlp(instance_repr)
        So center_mlp should start near zero → 1e-2 scale on last layer.
        At init: center ≈ anchor (the grid), which gives diverse predictions.

        Size and rotation MLPs also use 1e-2 so initial boxes are unit cubes
        with identity rotation at each anchor location.
        """
        for mlp in (self.center_mlp, self.size_mlp, self.rot_mlp):
            last = mlp.net[-1]
            last.weight.data.mul_(1e-2)
            nn.init.zeros_(last.bias)
        # Class head: default init + zero bias (scores ~0.5 at init)
        nn.init.zeros_(self.class_head.bias)

    def forward(self, instance_repr: torch.Tensor):
        """
        Args:
            instance_repr: [B, O, 256] from InstanceGatherer.

        Returns:
            Dict with:
                'center':  [B, O, 3]        anchor + residual
                'size':    [B, O, 3]         half-extent, exp-transformed, > 0
                'rot6d':   [B, O, 6]         raw 6D rotation (for debugging)
                'R':       [B, O, 3, 3]      proper rotation matrix (det=+1)
                'logits':  [B, O, 2]         dim 0=bg, dim 1=fg
                'corners': [B, O, 8, 3]      derived 3D box corners
        """
        B, O, _ = instance_repr.shape
        assert instance_repr.shape[-1] == self.in_dim

        # ── Inject per-query anchor into instance_repr ──
        # CRITICAL (Phase 3+ overfit debug, 4th iteration): even with the
        # per-query anchor as a fixed buffer, all queries collapse because
        # the InstanceGatherer outputs nearly-identical instance_repr for
        # all 64 queries (the gather operation homogenizes them when their
        # activation maps are similar). Without per-query identity in the
        # MLP input, the shared MLP produces identical residuals and the
        # only diversity comes from the (additive) anchor.
        #
        # Fix: project the per-query anchor to in_dim and add it to the
        # instance_repr BEFORE the MLPs. This guarantees each query has a
        # unique input feature, so the MLPs can produce per-query residuals.
        anchor = self.query_anchor[:O]                                  # [O, 3]
        anchor_feat = self.anchor_proj(anchor).unsqueeze(0).expand(B, -1, -1)  # [B, O, in_dim]
        # Use a residual connection so instance_repr's content is preserved.
        repr_with_anchor = instance_repr + anchor_feat                  # [B, O, in_dim]

        # Center = per-query spatial anchor + residual from MLP.
        center_residual = self.center_mlp(repr_with_anchor)             # [B, O, 3]
        anchor_b = anchor.unsqueeze(0).expand(B, -1, -1)                # [B, O, 3]
        center = anchor_b + center_residual                              # [B, O, 3]

        log_size = self.size_mlp(repr_with_anchor)                       # [B, O, 3]
        size = torch.exp(log_size.clamp(max=5.0))                        # [B, O, 3]
        rot6d = self.rot_mlp(repr_with_anchor)                           # [B, O, 6]
        logits = self.class_head(repr_with_anchor)                       # [B, O, 2]

        # 6D rotation → rotation matrix via Gram-Schmidt.
        identity_bias = rot6d.new_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        R = rotation_6d_to_matrix(rot6d + identity_bias)                 # [B, O, 3, 3]

        corners = _center_size_R_to_corners(center, size, R)             # [B, O, 8, 3]

        return {
            "center": center,
            "size": size,
            "rot6d": rot6d,
            "R": R,
            "logits": logits,
            "corners": corners,
        }
