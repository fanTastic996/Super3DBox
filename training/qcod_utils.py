"""
QCOD utility functions for Phase 2 loss computation.

Contains:
  - rotation_6d_to_matrix: 6D rotation → SO(3) via Gram-Schmidt
  - matrix_geodesic_loss: geodesic distance between rotation matrices
  - project_scene_corners_to_frames: scene GT projection (坐标契约 enforced)
  - build_token_activation_gt_from_projection: token-level binary mask (z>0 only)
  - build_fada_visibility_from_projection: frame-level area-weighted distribution (z>0 only)

All functions operate on already-normalized data (see plan D3 Step E docstring contracts).
"""

import torch
import torch.nn.functional as F

# =============================================================================
# Rotation utilities
# =============================================================================

def rotation_6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix via Gram-Schmidt.

    Reference: Zhou et al., "On the Continuity of Rotation Representations in
    Neural Networks", CVPR 2019.

    Args:
        rot6d: (..., 6) — two 3D column vectors concatenated.

    Returns:
        R: (..., 3, 3) proper rotation matrix (det = +1).
    """
    a1 = rot6d[..., :3]  # (..., 3)
    a2 = rot6d[..., 3:]  # (..., 3)

    # Gram-Schmidt: orthonormalize
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3) columns = b1,b2,b3


def matrix_geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    """Geodesic distance between two rotation matrices on SO(3).

    d(R1, R2) = arccos( (trace(R1^T R2) - 1) / 2 )

    Args:
        R_pred: (..., 3, 3) predicted rotation matrices.
        R_gt:   (..., 3, 3) ground-truth rotation matrices.
        eps: clamp margin for numerical stability.

    Returns:
        Scalar: mean geodesic distance in radians.
    """
    # R_rel = R_gt^T @ R_pred
    R_rel = R_gt.transpose(-1, -2) @ R_pred  # (..., 3, 3)
    # trace
    tr = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (...)
    # clamp for arccos stability
    cos_angle = ((tr - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos_angle)  # (...) in [0, pi]
    return angle.mean()


# =============================================================================
# Projection utilities
# =============================================================================

def project_scene_corners_to_frames(
    scene_corners: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    H_img: int,
    W_img: int,
):
    """Project scene GT corners (in sampled first-frame camera system) onto all frames.

    COORDINATE CONTRACT (any caller must obey; see plan D3 Step E):
    ──────────────────────────────────────────────────────────────────
    * scene_corners: already passed through load_scene_gt_in_first_frame()
      (world → sampled-first-frame camera) AND normalize_..._batch() (/ avg_scale).
      i.e., this is batch['qcod_scene_corners'] AFTER trainer._process_batch().
    * intrinsics: batch-internal camera intrinsics, already synchronized with
      BaseDataset.process_one_image() crop/resize/rotation augmentations.
    * extrinsics: from normalize_..._batch(). extrinsics[:, 0] = identity
      (first frame = reference); others are relative transforms.
      Translation component already divided by avg_scale.
    * THIS FUNCTION DOES NOT TOUCH all_poses.npy OR ANY UN-NORMALIZED DATA.
    ──────────────────────────────────────────────────────────────────

    Args:
        scene_corners: [B, N, 8, 3] — sampled-first-frame camera + normalized.
        intrinsics:    [B, S, 3, 3]
        extrinsics:    [B, S, 3, 4] — first frame = identity after normalization.
        H_img, W_img:  current batch image dimensions (dynamic, NOT fixed 518).

    Returns:
        uv:         [B, S, N, 8, 2] pixel coords. Corners behind the camera
                    (z <= eps) are filled with NaN — callers must use front_mask.
        front_mask: [B, S, N, 8] bool, True where z > eps (corner in front of camera).
        visible:    [B, S, N] bool, True if >= 4 front corners fall inside the image.
    """
    B, N, _, _ = scene_corners.shape
    S = extrinsics.shape[1]
    eps = 1e-3

    # 1. Transform scene corners (first-frame camera) → each frame's camera system.
    R = extrinsics[:, :, :3, :3]  # [B, S, 3, 3]
    t = extrinsics[:, :, :3, 3]   # [B, S, 3]
    # [B, 1, N, 8, 3] @ [B, S, 1, 1, 3, 3]^T + [B, S, 1, 1, 3]
    corners_in_frame = (
        torch.einsum("bnki,bsji->bsnkj", scene_corners, R)
        + t[:, :, None, None, :]
    )  # [B, S, N, 8, 3]

    # 2. Perspective projection.
    z = corners_in_frame[..., 2]  # [B, S, N, 8]
    front_mask = z > eps          # [B, S, N, 8]

    # Project: uv_h = K @ X_cam → divide by z
    # K: [B, S, 3, 3], corners_in_frame: [B, S, N, 8, 3]
    proj = torch.einsum("bsij,bsnkj->bsnki", intrinsics, corners_in_frame)
    # [B, S, N, 8, 3]

    # Safe division: only where z > eps; fill NaN elsewhere to force callers to mask.
    z_safe = torch.where(front_mask, z, torch.ones_like(z))
    uv = proj[..., :2] / z_safe.unsqueeze(-1)  # [B, S, N, 8, 2]
    uv = torch.where(front_mask.unsqueeze(-1), uv, torch.full_like(uv, float("nan")))

    # 3. Visibility: front + within image bounds.
    in_img = (
        front_mask
        & (uv[..., 0] >= 0)
        & (uv[..., 0] < W_img)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < H_img)
    )  # [B, S, N, 8]
    visible = in_img.sum(dim=-1) >= 4  # [B, S, N]

    return uv, front_mask, visible


def build_token_activation_gt_from_projection(
    projected_uv: torch.Tensor,
    front_mask: torch.Tensor,
    visible_mask: torch.Tensor,
    H_patch: int,
    W_patch: int,
    H_img: int,
    W_img: int,
) -> torch.Tensor:
    """Build per-instance token-level binary mask from projected 2D corners.

    IMPORTANT: Only uses corners where z > 0 (front_mask=True) to compute the
    2D bounding box. If all 8 corners are naively included, those behind the
    camera would have clamped/NaN uv values that blow the bbox to the full image,
    creating degenerate all-positive masks.

    Args:
        projected_uv:  [B, S, N, 8, 2] — NaN where z <= 0.
        front_mask:    [B, S, N, 8] bool.
        visible_mask:  [B, S, N] bool.
        H_patch, W_patch: dynamic patch grid size (H//14, W//14).
        H_img, W_img: current image dimensions.

    Returns:
        token_mask: [B, N, S, H_patch, W_patch] float (0/1).
                    Note: transposed to [B, N, S, ...] for easier per-instance indexing.
    """
    B, S, N = visible_mask.shape
    INF = float("inf")

    # Only consider front corners for bbox min/max.
    uv_for_min = torch.where(
        front_mask.unsqueeze(-1), projected_uv, torch.full_like(projected_uv, INF)
    )
    uv_for_max = torch.where(
        front_mask.unsqueeze(-1), projected_uv, torch.full_like(projected_uv, -INF)
    )

    uv_min = uv_for_min.amin(dim=-2)  # [B, S, N, 2]
    uv_max = uv_for_max.amax(dim=-2)  # [B, S, N, 2]

    # Clamp to image bounds.
    u_min = uv_min[..., 0].clamp(0, W_img)
    v_min = uv_min[..., 1].clamp(0, H_img)
    u_max = uv_max[..., 0].clamp(0, W_img)
    v_max = uv_max[..., 1].clamp(0, H_img)

    # Pixel → patch grid.
    sw = W_patch / W_img
    sh = H_patch / H_img
    px1 = (u_min * sw).clamp(0, W_patch - 1).long()
    py1 = (v_min * sh).clamp(0, H_patch - 1).long()
    px2 = (u_max * sw).ceil().clamp(0, W_patch).long()
    py2 = (v_max * sh).ceil().clamp(0, H_patch).long()

    # Build mask (loop is fine — N and S are small).
    device = projected_uv.device
    token_mask = torch.zeros(B, S, N, H_patch, W_patch, device=device)
    for b in range(B):
        for s in range(S):
            for n in range(N):
                if not visible_mask[b, s, n]:
                    continue
                y1 = int(py1[b, s, n])
                y2 = int(py2[b, s, n])
                x1 = int(px1[b, s, n])
                x2 = int(px2[b, s, n])
                if y2 > y1 and x2 > x1:
                    token_mask[b, s, n, y1:y2, x1:x2] = 1.0

    # Transpose to [B, N, S, H_patch, W_patch] for per-instance indexing.
    return token_mask.permute(0, 2, 1, 3, 4)


def build_fada_visibility_from_projection(
    projected_uv: torch.Tensor,
    front_mask: torch.Tensor,
    visible_mask: torch.Tensor,
    H_img: int,
    W_img: int,
) -> torch.Tensor:
    """Build FADA frame-level visibility distribution from projected 2D corners.

    Uses only z > 0 corners (via front_mask) to compute bbox area.
    Returns area-weighted, row-normalized probability distribution.

    Args:
        projected_uv:  [B, S, N, 8, 2]
        front_mask:    [B, S, N, 8]
        visible_mask:  [B, S, N]
        H_img, W_img: image dims.

    Returns:
        visibility: [B, N, S] float — each row sums to 1 (or 0 if never visible).
    """
    INF = float("inf")
    uv_for_min = torch.where(
        front_mask.unsqueeze(-1), projected_uv, torch.full_like(projected_uv, INF)
    )
    uv_for_max = torch.where(
        front_mask.unsqueeze(-1), projected_uv, torch.full_like(projected_uv, -INF)
    )
    uv_min = uv_for_min.amin(dim=-2)  # [B, S, N, 2]
    uv_max = uv_for_max.amax(dim=-2)

    u_min = uv_min[..., 0].clamp(0, W_img)
    v_min = uv_min[..., 1].clamp(0, H_img)
    u_max = uv_max[..., 0].clamp(0, W_img)
    v_max = uv_max[..., 1].clamp(0, H_img)

    w = (u_max - u_min).clamp(min=0)  # [B, S, N]
    h = (v_max - v_min).clamp(min=0)
    area = w * h * visible_mask.float()  # invisible → 0

    # Transpose to [B, N, S] and row-normalize.
    area = area.permute(0, 2, 1)  # [B, N, S]
    row_sum = area.sum(dim=-1, keepdim=True)
    visibility = torch.where(
        row_sum > 0, area / row_sum.clamp(min=1e-8), torch.zeros_like(area)
    )
    return visibility


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jensen-Shannon divergence between two distributions (last dim).

    Args:
        p, q: (..., D) non-negative, each row sums to ~1.
        eps: smoothing to avoid log(0).

    Returns:
        (...,) JS divergence per row.
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)
