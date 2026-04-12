"""
QCOD loss computation for Phase 2.

Implements compute_qcod_loss() which handles:
  - Stage A (default): chamfer + center_L1 + classification + rotation + size
  - Stage B: + activation BCE (when cfg.activation > 0)
  - Stage C: + FADA JS divergence (when cfg.fada > 0 and attn weights provided)

DESIGN NOTE (plan Step 2.2):
  * AABB-GIoU surrogate is ONLY used in the Hungarian matching cost.
    It is NOT a loss term. This is deliberate (not a bug). See plan D1/Step 2.2.
  * The existing pairwise_giou_3d_from_corners() is @torch.no_grad() — not differentiable.

IMPORT NOTE (plan Phase 2 bugfix #1):
  * This file does NOT import from loss.py at module level to avoid circular imports
    when loss.py later imports compute_qcod_loss. Instead, the five reused functions
    are imported lazily inside compute_qcod_loss() on first call.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from qcod_utils import (
    matrix_geodesic_loss,
    project_scene_corners_to_frames,
    build_token_activation_gt_from_projection,
    build_fada_visibility_from_projection,
    js_divergence,
)

# ── Lazy imports from loss.py (avoid circular dependency) ──────────────
# These are populated on first call to compute_qcod_loss().
_loss_fns = {}


def _ensure_loss_imports():
    """Lazy-import the five functions we reuse from loss.py.

    This breaks the potential loss.py -> qcod_loss.py -> loss.py cycle
    that would occur if MultitaskLoss (in loss.py) imports compute_qcod_loss.
    """
    if _loss_fns:
        return
    from loss import (
        build_3d_cost_logits,
        chamfer_loss,
        hungarian_2d_matching,
        objectness_bce_balanced_hardneg,
        _make_binary_labels,
    )
    _loss_fns["build_3d_cost_logits"] = build_3d_cost_logits
    _loss_fns["chamfer_loss"] = chamfer_loss
    _loss_fns["hungarian_2d_matching"] = hungarian_2d_matching
    _loss_fns["objectness_bce_balanced_hardneg"] = objectness_bce_balanced_hardneg
    _loss_fns["_make_binary_labels"] = _make_binary_labels


@dataclass
class QCODLossConfig:
    """Stage-aware loss weight configuration (see plan Phase 3+ bring-up table)."""
    chamfer: float = 1.0
    center_l1: float = 3.0
    cls: float = 1.0
    rotation: float = 1.0
    size: float = 0.3
    activation: float = 0.0    # Stage A: off; Stage B: 0.5
    fada: float = 0.0          # Stage A/B: off; Stage C: 0.3

    # Hungarian matching cost weights (separate from loss weights)
    match_chamfer: float = 1.0
    match_cls: float = 2.0
    match_giou: float = 1.5


def compute_qcod_loss(
    predictions: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    cfg: QCODLossConfig = None,
    patch_size: int = 14,
) -> Dict[str, torch.Tensor]:
    """Compute QCOD loss for one training step.

    Args:
        predictions: dict from QCODHead (or mock), expected keys:
            - 'qcod_corners':  [B, O, 8, 3]
            - 'qcod_center':   [B, O, 3]
            - 'qcod_size':     [B, O, 3]   (half-extent, exp'd)
            - 'qcod_logits':   [B, O, 2]   dim 0 = background, dim 1 = foreground
            - 'qcod_R':        [B, O, 3, 3]
            Required when cfg.activation > 0:
            - 'qcod_activation_logits': [B, O, S*P_patch]
            Required when cfg.fada > 0:
            - 'qcod_cross_attn_weights': List[Tensor], each [B, heads, O, S*P_patch]
              Must be non-empty (at least 1 layer's weights).
        batch: dict from trainer._process_batch(), expected keys:
            - 'qcod_scene_corners': [B, N_max, 8, 3]  (normalized, first-frame camera)
            - 'qcod_scene_R':       [B, N_max, 3, 3]
            - 'qcod_scene_scale':   [B, N_max, 3]     (half-extent, normalized)
            - 'qcod_scene_valid':   [B, N_max] bool
            - 'intrinsics':         [B, S, 3, 3]
            - 'extrinsics':         [B, S, 3, 4]
            - 'images':             [B, S, 3, H, W]    (for dynamic H/W)
        cfg: loss weight config.
        patch_size: ViT patch size (default 14), used to compute H_patch/W_patch.

    Returns:
        Dict with scalar tensors: 'total', 'chamfer', 'center', 'cls', 'rotation',
        'size', 'activation', 'fada', and 'num_matched' (int, for logging).

    Raises:
        KeyError: if cfg.activation > 0 but 'qcod_activation_logits' missing from predictions.
        KeyError: if cfg.fada > 0 but 'qcod_cross_attn_weights' missing from predictions.
        RuntimeError: if cfg.fada > 0 and 'qcod_cross_attn_weights' is present but empty list.
    """
    _ensure_loss_imports()
    build_3d_cost_logits = _loss_fns["build_3d_cost_logits"]
    chamfer_loss = _loss_fns["chamfer_loss"]
    hungarian_2d_matching = _loss_fns["hungarian_2d_matching"]
    objectness_bce_balanced_hardneg = _loss_fns["objectness_bce_balanced_hardneg"]
    _make_binary_labels = _loss_fns["_make_binary_labels"]

    if cfg is None:
        cfg = QCODLossConfig()

    # ── Fail-loud validation for activation / FADA config ──────────────
    # If a loss weight is > 0 but the corresponding prediction key is missing,
    # that is ALWAYS a bug (misconfigured pipeline), not a "graceful skip".
    if cfg.activation > 0:
        if "qcod_activation_logits" not in predictions:
            raise KeyError(
                "cfg.activation > 0 but 'qcod_activation_logits' not in predictions. "
                "This means the head is not producing activation logits, or the "
                "config/pipeline is misconfigured. Fix the head or set cfg.activation=0."
            )
    if cfg.fada > 0:
        if "qcod_cross_attn_weights" not in predictions:
            raise KeyError(
                "cfg.fada > 0 but 'qcod_cross_attn_weights' not in predictions. "
                "Set fada_layers != 'none' in QueryEvolutionModule, or set cfg.fada=0."
            )
        if len(predictions["qcod_cross_attn_weights"]) == 0:
            raise RuntimeError(
                "cfg.fada > 0 but qcod_cross_attn_weights is an empty list. "
                "This means fada_layers='none' in QueryEvolutionModule but cfg.fada > 0. "
                "Either set fada_layers='last' or set cfg.fada=0."
            )

    device = predictions["qcod_corners"].device
    B = predictions["qcod_corners"].shape[0]
    O = predictions["qcod_corners"].shape[1]

    # Image dimensions (dynamic, not hardcoded).
    H_img = batch["images"].shape[-2]
    W_img = batch["images"].shape[-1]
    S = batch["images"].shape[1]
    H_patch = H_img // patch_size
    W_patch = W_img // patch_size

    zero = torch.tensor(0.0, device=device, requires_grad=True)

    # Accumulate per-batch losses then average.
    loss_chamfer_sum = zero
    loss_center_sum = zero
    loss_cls_sum = zero
    loss_rot_sum = zero
    loss_size_sum = zero
    loss_act_sum = zero
    loss_fada_sum = zero
    total_matched = 0
    valid_batch_count = 0

    for b in range(B):
        # Extract valid GT for this sample.
        valid_mask = batch["qcod_scene_valid"][b]  # [N_max]
        gt_corners = batch["qcod_scene_corners"][b][valid_mask]  # [N_gt, 8, 3]
        gt_R = batch["qcod_scene_R"][b][valid_mask]              # [N_gt, 3, 3]
        gt_scale = batch["qcod_scene_scale"][b][valid_mask]      # [N_gt, 3]
        N_gt = gt_corners.shape[0]

        if N_gt == 0:
            continue

        pred_corners = predictions["qcod_corners"][b]  # [O, 8, 3]
        pred_logits = predictions["qcod_logits"][b]    # [O, 2]
        pred_center = predictions["qcod_center"][b]    # [O, 3]
        pred_size = predictions["qcod_size"][b]        # [O, 3]
        pred_R = predictions["qcod_R"][b]              # [O, 3, 3]

        # ── Hungarian Matching ──────────────────────────────────────────
        cost_kwargs = {
            "chamfer_weight": cfg.match_chamfer,
            "class_weight": cfg.match_cls,
            "giou_weight": cfg.match_giou,
        }
        cost_matrix = build_3d_cost_logits(
            pred_corners, gt_corners, pred_logits, cost_kwargs
        )  # [O, N_gt]
        pred_idx, gt_idx = hungarian_2d_matching(cost_matrix)
        # hungarian_2d_matching returns CPU tensors; move to device for indexing.
        num_matched = pred_idx.shape[0]
        total_matched += num_matched

        if num_matched == 0:
            continue

        valid_batch_count += 1
        pred_idx = pred_idx.to(device)
        gt_idx = gt_idx.to(device)

        matched_pred_corners = pred_corners[pred_idx]  # [M, 8, 3]
        matched_gt_corners = gt_corners[gt_idx]        # [M, 8, 3]

        # ── Chamfer Loss ────────────────────────────────────────────────
        l_chamfer = chamfer_loss(matched_pred_corners, matched_gt_corners)

        # ── Center L1 ──────────────────────────────────────────────────
        pred_center_matched = pred_center[pred_idx]
        gt_center_matched = matched_gt_corners.mean(dim=1)  # [M, 3]
        l_center = F.l1_loss(pred_center_matched, gt_center_matched)

        # ── Classification (BCE + hard negative mining) ────────────────
        # qcod_logits layout: dim 0 = background, dim 1 = foreground.
        # We take dim 1 as the foreground logit for BCE.
        fg_logits = pred_logits[:, 1]  # [O] foreground logit
        cls_targets = _make_binary_labels(O, pred_idx, device)
        l_cls, _ = objectness_bce_balanced_hardneg(fg_logits, cls_targets)

        # ── Rotation (geodesic on SO(3)) ───────────────────────────────
        matched_pred_R = pred_R[pred_idx]    # [M, 3, 3]
        matched_gt_R = gt_R[gt_idx]          # [M, 3, 3]
        l_rot = matrix_geodesic_loss(matched_pred_R, matched_gt_R)

        # ── Size (volume-weighted L1 on log half-extent) ──────────────
        # gt_scale is already half-extent (loader did / 2) and normalized (/ avg_scale).
        # pred_size is exp(log_size), also half-extent.
        # DO NOT divide gt_scale by 2 again here (plan D1 errata).
        #
        # Volume-weighting (plan Stage A+ diagnosis): L1(log) treats small and
        # large objects equally in log-space, but the gradient inherently favors
        # small objects (a 3x error on a tiny object gives more log-loss than a
        # 2x error on a large object). We re-weight each matched pair by
        # max(gt_volume / median_gt_volume, 1.0) so large objects get
        # proportionally stronger size gradients.
        matched_pred_size = pred_size[pred_idx]  # [M, 3]
        matched_gt_size = gt_scale[gt_idx]       # [M, 3]

        log_pred = torch.log(matched_pred_size.clamp(min=1e-6))
        log_gt = torch.log(matched_gt_size.clamp(min=1e-6))

        # Per-instance volume weight: product of 3 half-extents as proxy
        gt_vol = matched_gt_size.prod(dim=-1)          # [M]
        median_vol = gt_vol.median().clamp(min=1e-10)
        size_weight = (gt_vol / median_vol).clamp(min=1.0)  # [M], ≥1.0
        # Per-dim L1, then weighted mean across instances
        per_instance_size_loss = (log_pred - log_gt).abs().mean(dim=-1)  # [M]
        l_size = (per_instance_size_loss * size_weight).sum() / size_weight.sum()

        # ── Accumulate core losses ─────────────────────────────────────
        loss_chamfer_sum = loss_chamfer_sum + l_chamfer
        loss_center_sum = loss_center_sum + l_center
        loss_cls_sum = loss_cls_sum + l_cls
        loss_rot_sum = loss_rot_sum + l_rot
        loss_size_sum = loss_size_sum + l_size

        # ── Activation (Stage B, default off) ──────────────────────────
        if cfg.activation > 0:
            # Key existence already validated above (fail-loud).
            act_logits_b = predictions["qcod_activation_logits"][b]  # [O, S*P_patch]
            # Build token activation GT from projected scene corners.
            uv, front_mask, visible = project_scene_corners_to_frames(
                batch["qcod_scene_corners"][b:b+1],  # [1, N_max, 8, 3]
                batch["intrinsics"][b:b+1],
                batch["extrinsics"][b:b+1],
                H_img, W_img,
            )
            token_mask_gt = build_token_activation_gt_from_projection(
                uv, front_mask, visible, H_patch, W_patch, H_img, W_img
            )  # [1, N_max, S, H_patch, W_patch]
            # Index matched GT instances. token_mask_gt is on the same device
            # as the projection inputs (which follow the batch device).
            gt_act = token_mask_gt[0, gt_idx]  # [M, S, H_patch, W_patch]
            gt_act_flat = gt_act.reshape(num_matched, S * H_patch * W_patch)
            pred_act_matched = act_logits_b[pred_idx]  # [M, S*P_patch]
            l_act = F.binary_cross_entropy_with_logits(
                pred_act_matched, gt_act_flat, reduction="mean"
            )
            loss_act_sum = loss_act_sum + l_act

        # ── FADA (Stage C, default off) ────────────────────────────────
        if cfg.fada > 0:
            # Key existence + non-empty already validated above (fail-loud).
            attn_weights_list = predictions["qcod_cross_attn_weights"]
            # Average across saved layers.
            # Each element: [B, heads, O, S*P_patch]
            attn_stack = torch.stack(
                [aw[b] for aw in attn_weights_list], dim=0
            )  # [L, heads, O, S*P_patch]
            attn_avg = attn_stack.mean(dim=(0, 1))  # [O, S*P_patch]
            # Marginalize to frame level: [O, S]
            P_patch = H_patch * W_patch
            attn_frame = attn_avg.view(O, S, P_patch).sum(dim=-1)  # [O, S]
            attn_frame = attn_frame / attn_frame.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Build FADA GT from projection.
            uv, front_mask, visible = project_scene_corners_to_frames(
                batch["qcod_scene_corners"][b:b+1],
                batch["intrinsics"][b:b+1],
                batch["extrinsics"][b:b+1],
                H_img, W_img,
            )
            vis_gt = build_fada_visibility_from_projection(
                uv, front_mask, visible, H_img, W_img
            )  # [1, N_max, S]
            # Index matched GT instances (on device, no .cpu() needed).
            matched_vis_gt = vis_gt[0, gt_idx]  # [M, S]
            matched_attn = attn_frame[pred_idx]  # [M, S]
            l_fada = js_divergence(matched_attn, matched_vis_gt).mean()
            loss_fada_sum = loss_fada_sum + l_fada

    # ── Average and weight ──────────────────────────────────────────────
    denom = max(valid_batch_count, 1)
    losses = {
        "chamfer": loss_chamfer_sum / denom,
        "center": loss_center_sum / denom,
        "cls": loss_cls_sum / denom,
        "rotation": loss_rot_sum / denom,
        "size": loss_size_sum / denom,
        "activation": loss_act_sum / denom,
        "fada": loss_fada_sum / denom,
    }

    total = (
        cfg.chamfer * losses["chamfer"]
        + cfg.center_l1 * losses["center"]
        + cfg.cls * losses["cls"]
        + cfg.rotation * losses["rotation"]
        + cfg.size * losses["size"]
        + cfg.activation * losses["activation"]
        + cfg.fada * losses["fada"]
    )
    losses["total"] = total
    losses["num_matched"] = total_matched

    return losses
