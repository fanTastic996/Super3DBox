"""
Enhanced Slot Attention for Multi-View Instance Aggregation

Core module for discovering object instances in VGGT's multi-view aligned
feature space via competitive attention with mask-guided refinement.

Key components:
  - MultiScaleFeatureProjector: fuse multi-layer VGGT features
  - PositionEmbedding: 2D sinusoidal + learnable frame embedding
  - EnhancedSlotAttention: competitive softmax + mask-guided + GRU
  - SlotObjectnessHead: per-slot objectness → token-level predictions
  - GTAnchoredContrastive: view-invariance regularization
  - create_complementary_view_masks: masking for contrastive path
  - project_boxes_to_instance_labels: GT token-level instance labels
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def world_to_camera(points, pose):
    """
    Transform world-coordinate points to camera coordinates.

    IMPORTANT: all_poses.npy stores camera-to-world [R|t], where t is camera
    center in world coords. The correct world-to-camera transform is:
        p_cam = R^T @ (p_world - t)

    Args:
        points: [..., 3] world coordinates
        pose: [4, 4] camera-to-world transform

    Returns:
        points_cam: [..., 3] camera coordinates
    """
    R = pose[:3, :3]  # camera-to-world rotation
    t = pose[:3, 3]   # camera center in world coords
    # world-to-camera: R^T @ (p - t)
    return (points - t) @ R  # equivalent to (R^T @ (p-t)^T)^T


# ─────────────────────────────────────────────
# Position Embedding
# ─────────────────────────────────────────────

def create_sinusoidal_2d(h, w, dim):
    """Create 2D sinusoidal position embedding [h*w, dim]."""
    assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
    quarter = dim // 4
    div_term = torch.exp(
        torch.arange(0, quarter, dtype=torch.float32)
        * -(math.log(10000.0) / quarter)
    )

    pos_h = torch.arange(h, dtype=torch.float32).unsqueeze(1)  # [h, 1]
    pos_w = torch.arange(w, dtype=torch.float32).unsqueeze(1)  # [w, 1]

    pe_h_sin = torch.sin(pos_h * div_term)  # [h, quarter]
    pe_h_cos = torch.cos(pos_h * div_term)
    pe_w_sin = torch.sin(pos_w * div_term)
    pe_w_cos = torch.cos(pos_w * div_term)

    # Interleave sin/cos: [h, dim//2] and [w, dim//2]
    pe_h = torch.zeros(h, dim // 2)
    pe_h[:, 0::2] = pe_h_sin
    pe_h[:, 1::2] = pe_h_cos
    pe_w = torch.zeros(w, dim // 2)
    pe_w[:, 0::2] = pe_w_sin
    pe_w[:, 1::2] = pe_w_cos

    # Outer product → [h*w, dim]
    pe = torch.zeros(h * w, dim)
    for i in range(h):
        pe[i * w : (i + 1) * w, : dim // 2] = pe_h[i].unsqueeze(0)
        pe[i * w : (i + 1) * w, dim // 2 :] = pe_w

    return pe


class PositionEmbedding(nn.Module):
    """2D sinusoidal position + learnable frame embedding."""

    def __init__(self, dim, grid_h=37, grid_w=37, max_views=16):
        super().__init__()
        pos_embed = create_sinusoidal_2d(grid_h, grid_w, dim)
        self.register_buffer("pos_embed", pos_embed)  # [H*W, D]
        self.frame_embed = nn.Embedding(max_views, dim)
        self.n_patches = grid_h * grid_w

    def forward(self, features, S):
        """
        Args:
            features: [B, S*n_patches, D]
            S: number of views
        Returns:
            features + positional + frame embeddings
        """
        B, N, D = features.shape
        device = features.device

        # 2D position: repeat for each view
        pos = self.pos_embed.unsqueeze(0).expand(S, -1, -1)  # [S, n_patches, D]
        pos = pos.reshape(1, S * self.n_patches, D).expand(B, -1, -1)

        # Frame index
        frame_idx = torch.arange(S, device=device).repeat_interleave(self.n_patches)
        frame_emb = self.frame_embed(frame_idx).unsqueeze(0).expand(B, -1, -1)

        return features + pos + frame_emb


# ─────────────────────────────────────────────
# Multi-Scale Feature Projector
# ─────────────────────────────────────────────

class MultiScaleFeatureProjector(nn.Module):
    """Project multi-layer VGGT features into unified detection space."""

    def __init__(self, in_dim=2048, out_dim=256, num_layers=4):
        super().__init__()
        per_layer_dim = out_dim // num_layers
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, per_layer_dim),
                nn.GELU(),
            )
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, multi_layer_features):
        """
        Args:
            multi_layer_features: list of [B, S, n_patches, 2048]
        Returns:
            [B, S, n_patches, out_dim]
        """
        projected = [proj(feat) for proj, feat in
                     zip(self.projections, multi_layer_features)]
        combined = torch.cat(projected, dim=-1)
        return self.out_norm(combined)


class SingleLayerProjector(nn.Module):
    """Simple projection for single-layer features (Step 1 validation)."""

    def __init__(self, in_dim=2048, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, features):
        """
        Args:
            features: [B, S, n_patches, in_dim]
        Returns:
            [B, S, n_patches, out_dim]
        """
        return self.proj(features)


# ─────────────────────────────────────────────
# Enhanced Slot Attention
# ─────────────────────────────────────────────

class EnhancedSlotAttention(nn.Module):
    """
    Enhanced Slot Attention with:
      1. Competitive softmax (normalize over slot dim)
      2. Mask-Guided iterative refinement
      3. GRU update + MLP residual
      4. Global slots exempt from mask
    """

    def __init__(
        self,
        dim=256,
        num_slots=32,
        num_global_slots=2,
        num_iterations=3,
        mlp_hidden_dim=512,
        max_mask_strength=0.6,
        eps=1e-8,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_global_slots = num_global_slots
        self.num_iterations = num_iterations
        self.max_mask_strength = max_mask_strength
        self.eps = eps

        # Slot initialization: K foreground + 1 background sink
        self.num_sink_slots = 1
        self.slot_embed = nn.Embedding(num_slots + self.num_sink_slots, dim)
        nn.init.normal_(self.slot_embed.weight, std=0.02)

        # Projections
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_q = nn.Linear(dim, dim, bias=False)

        # Learnable logit scale (replaces fixed D^{-0.5})
        self.logit_scale = nn.Parameter(torch.ones([]) * 5.0)

        # Learnable sink bias: positive bias makes tokens default to sink slot
        self.sink_bias = nn.Parameter(torch.tensor(2.0))

        # GRU update
        self.gru = nn.GRUCell(dim, dim)

        # Residual gate for mask memory (prevents bad attention from locking next iteration)
        self.mask_gate = nn.Parameter(torch.tensor(0.5))

        # MLP residual
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def _initialize_slots(self, batch_size, device):
        """Return fixed learned slot embeddings (no random sampling)."""
        return self.slot_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, features, view_mask=None, slots_init=None, temperature=1.0):
        """
        Args:
            features: [B, N, D] where N = S * n_patches
            view_mask: [B, N] binary, 1=visible 0=masked (for contrastive path)
            slots_init: [B, K, D] optional pre-initialized slots
            temperature: float, softmax temperature (< 1 = sharper competition)
        Returns:
            slots: [B, K, D] refined slot representations
            attn: [B, K, N] final iteration attention maps
            all_attns: list of T attention maps
        """
        B, N, D = features.shape
        K_total = self.num_slots + self.num_sink_slots  # K foreground + 1 sink
        G = self.num_global_slots

        # Initialize slots
        if slots_init is not None:
            slots = slots_init
        else:
            slots = self._initialize_slots(B, features.device)

        # Pre-compute keys and values (shared across iterations)
        inputs_normed = self.norm_inputs(features)
        k = F.normalize(self.proj_k(inputs_normed), dim=-1)  # [B, N, D] L2-normed
        v = self.proj_v(inputs_normed)  # [B, N, D]

        prev_attn = None
        all_attns = []

        for t in range(self.num_iterations):
            slots_prev = slots

            # Query from current slots (L2-normed like keys)
            q = F.normalize(self.proj_q(self.norm_slots(slots)), dim=-1)  # [B, K, D]

            # Attention logits with clamped learnable scale
            scale = torch.clamp(self.logit_scale, min=0.1, max=10.0)
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, K, N]
            attn_logits = torch.clamp(attn_logits, min=-15.0, max=15.0)

            # Mask-Guided refinement with residual gate (from iteration 2 onwards)
            if prev_attn is not None and t > 0:
                strength = self.max_mask_strength * t / self.num_iterations
                log_attn = torch.log(prev_attn + self.eps)  # [B, K, N]
                log_attn_mean = log_attn.mean(dim=-1, keepdim=True)  # [B, K, 1]
                new_bias = strength * (log_attn - log_attn_mean)
                # Residual gate: blend new bias with accumulated prev_bias
                # gate → 1: trust new attention; gate → 0: rely on history
                gate = torch.sigmoid(self.mask_gate)
                if not hasattr(self, '_prev_bias') or self._prev_bias is None:
                    mask_bias = new_bias
                else:
                    mask_bias = gate * new_bias + (1 - gate) * self._prev_bias
                self._prev_bias = mask_bias.detach()
                # Global slots and sink slot exempt from mask bias
                mask_bias[:, :G, :] = 0.0
                mask_bias[:, -self.num_sink_slots:, :] = 0.0
                attn_logits = attn_logits + mask_bias
            else:
                self._prev_bias = None

            # Sink slot bias: tokens default to sink unless foreground slots claim them
            attn_logits[:, -self.num_sink_slots:, :] = attn_logits[:, -self.num_sink_slots:, :] + self.sink_bias

            # Competitive softmax: normalize over slot dimension
            attn = F.softmax(attn_logits / temperature, dim=1)  # [B, K, N]

            # Apply view mask AFTER softmax (zero out invisible tokens)
            if view_mask is not None:
                attn = attn * view_mask.unsqueeze(1)  # [B, K, N]

            # Save for next iteration (detached, no grad through mask)
            prev_attn = attn.detach()
            all_attns.append(attn)

            # Weighted aggregation (per-slot weighted mean)
            attn_sum = attn.sum(dim=-1, keepdim=True)  # [B, K, 1]
            updates = torch.bmm(attn, v)  # [B, K, D]
            updates = updates / (attn_sum + self.eps)

            # GRU update
            slots = self.gru(
                updates.reshape(B * K_total, D),
                slots_prev.reshape(B * K_total, D),
            ).reshape(B, K_total, D)

            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn, all_attns


# ─────────────────────────────────────────────
# Slot Objectness Head
# ─────────────────────────────────────────────

class SlotObjectnessHead(nn.Module):
    """Per-slot objectness classifier → token-level objectness prediction."""

    def __init__(self, dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, slots, attn_maps):
        """
        Args:
            slots: [B, K, D]
            attn_maps: [B, K, N] competitive attention maps
        Returns:
            token_objectness: [B, N] predicted objectness per token
            slot_objectness: [B, K] per-slot objectness logits
        """
        slot_logits = self.head(slots).squeeze(-1)  # [B, K]
        slot_probs = torch.sigmoid(slot_logits)  # [B, K]

        # Token objectness = weighted sum of slot objectness
        # Each token's objectness = sum over slots of (slot_prob * attention)
        token_obj = torch.einsum("bk,bkn->bn", slot_probs, attn_maps)  # [B, N]

        return token_obj, slot_logits


# ─────────────────────────────────────────────
# Co-Visibility Camera Selection
# ─────────────────────────────────────────────

def compute_visibility_matrix(bbox_corners, all_poses, K_intrinsic, img_h, img_w):
    """
    Compute which boxes are visible from which cameras.

    Args:
        bbox_corners: [N_box, 8, 3] world coordinates
        all_poses: [N_cam, 4, 4] world-to-camera transforms
        K_intrinsic: [3, 3] camera intrinsics
        img_h, img_w: original image dimensions

    Returns:
        visibility: [N_cam, N_box] boolean matrix
    """
    N_cam = len(all_poses)
    N_box = len(bbox_corners)

    R_all = all_poses[:, :3, :3]  # [N_cam, 3, 3] camera-to-world
    t_all = all_poses[:, :3, 3]   # [N_cam, 3] camera center in world

    # world-to-camera: p_cam = R^T @ (p_world - t)
    # = (p_world - t) @ R  (transposed form for batch)
    # bbox_corners: [N_box, 8, 3], t_all: [N_cam, 3]
    shifted = bbox_corners[np.newaxis, :, :, :] - t_all[:, np.newaxis, np.newaxis, :]
    # shifted: [N_cam, N_box, 8, 3]
    # corners_cam[c,b,p,:] = shifted[c,b,p,:] @ R_all[c]
    corners_cam = np.einsum("cbpi,cij->cbpj", shifted, R_all)

    # Check z > 0.1 for at least 4 corners
    valid_z = corners_cam[:, :, :, 2] > 0.1   # [N_cam, N_box, 8]
    enough_valid = valid_z.sum(axis=2) >= 4    # [N_cam, N_box]

    # Project to 2D (safe division)
    z_safe = np.maximum(corners_cam[:, :, :, 2], 0.1)
    fx, fy = K_intrinsic[0, 0], K_intrinsic[1, 1]
    cx, cy = K_intrinsic[0, 2], K_intrinsic[1, 2]
    px = fx * corners_cam[:, :, :, 0] / z_safe + cx  # [N_cam, N_box, 8]
    py = fy * corners_cam[:, :, :, 1] / z_safe + cy

    # Check if projected bounding rect overlaps image
    px_min = px.min(axis=2)  # [N_cam, N_box]
    px_max = px.max(axis=2)
    py_min = py.min(axis=2)
    py_max = py.max(axis=2)

    in_image = (px_max > 0) & (px_min < img_w) & (py_max > 0) & (py_min < img_h)

    # Also check that projected box isn't too tiny (< 2 patches)
    patch_w = img_w / 37.0
    proj_w = np.clip(px_max, 0, img_w) - np.clip(px_min, 0, img_w)
    proj_h = np.clip(py_max, 0, img_h) - np.clip(py_min, 0, img_h)
    big_enough = (proj_w > patch_w) & (proj_h > patch_w)

    visibility = enough_valid & in_image & big_enough
    return visibility


def select_covisible_cameras(bbox_corners, all_poses, K_intrinsic, img_h, img_w,
                             num_cameras=4):
    """
    Select cameras with maximum co-visibility of 3D boxes.

    Strategy:
    1. Compute per-camera box visibility matrix
    2. Seed with camera seeing most boxes
    3. Greedily add cameras maximizing co-visibility with current set

    Returns:
        selected_indices: list of camera indices, or None if insufficient data
    """
    if len(bbox_corners) == 0 or len(all_poses) < num_cameras:
        return None

    visibility = compute_visibility_matrix(
        bbox_corners, all_poses, K_intrinsic, img_h, img_w
    )
    cam_scores = visibility.sum(axis=1)  # [N_cam]

    # Filter cameras that see at least 1 box
    valid_cams = np.where(cam_scores > 0)[0]
    if len(valid_cams) < num_cameras:
        return None

    # Greedy co-visibility selection
    selected = []
    seed = valid_cams[cam_scores[valid_cams].argmax()]
    selected.append(int(seed))

    for _ in range(num_cameras - 1):
        current_visible = visibility[selected].any(axis=0)  # union of visible boxes

        best_score = -1
        best_cam = -1

        for cam_idx in valid_cams:
            if cam_idx in selected:
                continue
            # Co-visibility: boxes visible from BOTH this camera AND current set
            co_vis = (visibility[cam_idx] & current_visible).sum()
            # Also value cameras that add new visible boxes
            new_boxes = (visibility[cam_idx] & ~current_visible).sum()
            score = 2.0 * co_vis + 0.5 * new_boxes
            if score > best_score:
                best_score = score
                best_cam = int(cam_idx)

        if best_cam >= 0:
            selected.append(best_cam)

    if len(selected) < num_cameras:
        return None

    return selected


# ─────────────────────────────────────────────
# View Masking for Contrastive Path
# ─────────────────────────────────────────────

def create_complementary_view_masks(S, n_patches=1369, mask_ratio=0.5, device="cpu"):
    """
    Create two complementary view masks for contrastive learning.
    Keeps tensor shape [S*n_patches] identical; uses binary masking.

    Returns:
        mask_a: [S*n_patches] binary, 1=visible 0=masked
        mask_b: [S*n_patches] binary, complementary to mask_a
    """
    n_mask = max(1, int(S * mask_ratio))
    n_mask = min(n_mask, S - 1)  # keep at least 1 view per group
    perm = torch.randperm(S)

    views_a = set(perm[:S - n_mask].tolist())
    views_b = set(perm[S - n_mask:].tolist())

    mask_a = torch.zeros(S * n_patches, device=device)
    mask_b = torch.zeros(S * n_patches, device=device)

    for v in range(S):
        start = v * n_patches
        end = (v + 1) * n_patches
        if v in views_a:
            mask_a[start:end] = 1.0
        if v in views_b:
            mask_b[start:end] = 1.0

    return mask_a, mask_b


# ─────────────────────────────────────────────
# GT-Anchored Contrastive Loss
# ─────────────────────────────────────────────

class GTAnchoredContrastive(nn.Module):
    """
    GT-Anchored Slot Contrastive Learning.

    Instead of matching slots across groups by feature similarity (cold-start problem),
    match each group's slots to GT instances via attention-affinity with fixed GT labels.
    """

    def __init__(self, temperature=0.07, activity_tau=0.03, activity_temp=0.015):
        super().__init__()
        self.temperature = temperature
        self.activity_tau = activity_tau
        self.activity_temp = activity_temp

    def compute_slot_instance_affinity(self, attn, instance_ids, num_instances):
        """
        Compute affinity between each slot and each GT instance.

        Args:
            attn: [B, K, N] slot attention maps
            instance_ids: [B, N] int, 0=bg, 1..M=instance
            num_instances: max number of instances

        Returns:
            affinity: [B, K, M] slot-instance affinity
        """
        B, K, N = attn.shape
        M = num_instances
        affinity = torch.zeros(B, K, M, device=attn.device)

        for j in range(1, M + 1):
            inst_mask = (instance_ids == j).float()  # [B, N]
            # How much of slot k's attention falls on instance j's tokens
            affinity[:, :, j - 1] = (attn * inst_mask.unsqueeze(1)).sum(-1)

        return affinity

    def match_slots_to_instances(self, affinity):
        """
        Hungarian matching: maximize affinity → each slot matches one instance.

        Args:
            affinity: [B, K, M]

        Returns:
            list of (slot_indices, instance_indices) per batch element
        """
        from scipy.optimize import linear_sum_assignment

        B = affinity.shape[0]
        matches = []
        cost = -affinity.detach().cpu().numpy()

        for b in range(B):
            row_ind, col_ind = linear_sum_assignment(cost[b])
            matches.append((
                torch.tensor(row_ind, dtype=torch.long),
                torch.tensor(col_ind, dtype=torch.long),
            ))

        return matches

    def compute_activity_weight(self, attn, slot_idx):
        """Soft sigmoid weight based on slot attention mass."""
        mass = attn[:, slot_idx, :].sum(dim=-1)  # [B]
        return torch.sigmoid((mass - self.activity_tau) / self.activity_temp)

    def forward(self, slots_a, slots_b, attn_a, attn_b, instance_ids):
        """
        Args:
            slots_a: [B, K, D] slot features from group A
            slots_b: [B, K, D] slot features from group B
            attn_a: [B, K, N] attention maps from group A
            attn_b: [B, K, N] attention maps from group B
            instance_ids: [B, N] GT instance labels (0=bg, 1..M)

        Returns:
            loss: scalar contrastive loss
            info: dict with diagnostics
        """
        B, K, D = slots_a.shape
        num_instances = int(instance_ids.max().item())

        if num_instances == 0:
            return torch.tensor(0.0, device=slots_a.device), {"n_pairs": 0}

        # Compute affinity for both groups
        affinity_a = self.compute_slot_instance_affinity(attn_a, instance_ids, num_instances)
        affinity_b = self.compute_slot_instance_affinity(attn_b, instance_ids, num_instances)

        # Match each group's slots to GT instances
        matches_a = self.match_slots_to_instances(affinity_a)
        matches_b = self.match_slots_to_instances(affinity_b)

        # Build cross-group pairs via shared instance ID
        total_loss = torch.tensor(0.0, device=slots_a.device)
        n_pairs = 0

        for b in range(B):
            slot_idx_a, inst_idx_a = matches_a[b]
            slot_idx_b, inst_idx_b = matches_b[b]

            # Build instance → slot mapping for group B
            inst_to_slot_b = {}
            for s, i in zip(slot_idx_b.tolist(), inst_idx_b.tolist()):
                inst_to_slot_b[i] = s

            # For each matched slot in group A, find the slot in group B
            # matched to the same instance
            anchors = []
            positives = []
            weights = []

            for s_a, i_a in zip(slot_idx_a.tolist(), inst_idx_a.tolist()):
                if i_a in inst_to_slot_b:
                    s_b = inst_to_slot_b[i_a]
                    w_a = self.compute_activity_weight(attn_a[b:b+1], s_a)
                    w_b = self.compute_activity_weight(attn_b[b:b+1], s_b)
                    pair_weight = (w_a * w_b).squeeze()

                    if pair_weight > 0.01:  # skip very inactive pairs
                        anchors.append(slots_a[b, s_a])
                        positives.append(slots_b[b, s_b])
                        weights.append(pair_weight)

            if len(anchors) == 0:
                continue

            anchors = torch.stack(anchors)      # [P, D]
            positives = torch.stack(positives)   # [P, D]
            pair_weights = torch.stack(weights)  # [P]

            # InfoNCE: each anchor's positive is its matched slot,
            # negatives are all other group B slots
            anchor_norm = F.normalize(anchors, dim=-1)
            positive_norm = F.normalize(positives, dim=-1)
            all_b_norm = F.normalize(slots_b[b], dim=-1)  # [K, D]

            # Similarity: anchor vs all group B slots
            sim = torch.mm(anchor_norm, all_b_norm.t()) / self.temperature  # [P, K]

            # Positive indices in all_b_norm
            pos_indices = torch.tensor(
                [inst_to_slot_b[i_a] for s_a, i_a in
                 zip(slot_idx_a.tolist(), inst_idx_a.tolist())
                 if i_a in inst_to_slot_b],
                device=slots_a.device, dtype=torch.long
            )[:len(anchors)]

            # InfoNCE loss per pair
            loss_per_pair = F.cross_entropy(sim, pos_indices, reduction="none")
            weighted_loss = (loss_per_pair * pair_weights).sum()
            total_loss = total_loss + weighted_loss
            n_pairs += len(anchors)

        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        return total_loss, {"n_pairs": n_pairs}


# ─────────────────────────────────────────────
# Token-Level GT Label Generation
# ─────────────────────────────────────────────

def project_boxes_to_instance_labels(
    bbox_corners, K_intrinsic, pose, img_h, img_w,
    grid_h=37, grid_w=37
):
    """
    Project 3D boxes to patch grid, returning per-instance labels.

    Args:
        bbox_corners: [N_box, 8, 3] world coordinates
        K_intrinsic: [3, 3] camera intrinsics
        pose: [4, 4] world-to-camera extrinsic
        img_h, img_w: original image dimensions
        grid_h, grid_w: patch grid dimensions

    Returns:
        instance_ids: [grid_h*grid_w] int64, 0=bg, 1..N=instance
    """
    n_tokens = grid_h * grid_w
    instance_ids = np.zeros(n_tokens, dtype=np.int64)
    instance_depths = np.full(n_tokens, np.inf, dtype=np.float32)

    if bbox_corners is None or len(bbox_corners) == 0:
        return instance_ids

    for box_idx, box in enumerate(bbox_corners):
        corners_cam = world_to_camera(box, pose)  # [8, 3]
        valid = corners_cam[:, 2] > 0.1
        if valid.sum() < 4:
            continue

        # Depth of box center in camera frame
        center_depth = corners_cam.mean(axis=0)[2]

        corners_valid = corners_cam[valid]
        fx, fy = K_intrinsic[0, 0], K_intrinsic[1, 1]
        cx, cy = K_intrinsic[0, 2], K_intrinsic[1, 2]
        px = fx * corners_valid[:, 0] / corners_valid[:, 2] + cx
        py = fy * corners_valid[:, 1] / corners_valid[:, 2] + cy

        x_min = max(0, int(np.floor(px.min())))
        x_max = min(img_w - 1, int(np.ceil(px.max())))
        y_min = max(0, int(np.floor(py.min())))
        y_max = min(img_h - 1, int(np.ceil(py.max())))

        if x_max <= x_min or y_max <= y_min:
            continue

        patch_x_min = max(0, x_min * grid_w // img_w)
        patch_x_max = min(grid_w - 1, x_max * grid_w // img_w)
        patch_y_min = max(0, y_min * grid_h // img_h)
        patch_y_max = min(grid_h - 1, y_max * grid_h // img_h)

        for py_idx in range(patch_y_min, patch_y_max + 1):
            for px_idx in range(patch_x_min, patch_x_max + 1):
                token_idx = py_idx * grid_w + px_idx
                # Depth ordering: closer box wins
                if center_depth < instance_depths[token_idx]:
                    instance_ids[token_idx] = box_idx + 1
                    instance_depths[token_idx] = center_depth

    return instance_ids


def project_boxes_to_objectness(bbox_corners, K_intrinsic, pose, img_h, img_w,
                                grid_h=37, grid_w=37):
    """Binary objectness labels (0=bg, 1=obj). Convenience wrapper."""
    inst = project_boxes_to_instance_labels(
        bbox_corners, K_intrinsic, pose, img_h, img_w, grid_h, grid_w
    )
    return (inst > 0).astype(np.float32)


# ─────────────────────────────────────────────
# Full ESA Model for Validation
# ─────────────────────────────────────────────

class ESAValidationModel(nn.Module):
    """
    Minimal model for Step 1 validation:
      SingleLayerProjector → PositionEmbedding → EnhancedSlotAttention → SlotObjectnessHead
    """

    def __init__(
        self,
        in_dim=2048,
        slot_dim=256,
        num_slots=32,
        num_global_slots=2,
        num_iterations=3,
        max_views=16,
    ):
        super().__init__()
        self.projector = SingleLayerProjector(in_dim, slot_dim)
        self.pos_embed = PositionEmbedding(slot_dim, max_views=max_views)
        self.slot_attention = EnhancedSlotAttention(
            dim=slot_dim,
            num_slots=num_slots,
            num_global_slots=num_global_slots,
            num_iterations=num_iterations,
        )
        self.objectness_head = SlotObjectnessHead(slot_dim)
        self.slot_dim = slot_dim
        self.num_slots = num_slots

    def forward(self, features, S, view_mask=None, temperature=1.0):
        """
        Args:
            features: [B, S, 1369, 2048] raw VGGT patch tokens
            S: number of views
            view_mask: [B, S*1369] optional binary mask
            temperature: float, softmax temperature

        Returns:
            slots: [B, K, D] slot features
            attn: [B, K, S*1369] final attention maps
            all_attns: list of attention maps per iteration
        """
        B = features.shape[0]

        # Project features
        projected = self.projector(features)  # [B, S, 1369, D]

        # Flatten views
        N = S * projected.shape[2]
        flat = projected.reshape(B, N, self.slot_dim)  # [B, S*1369, D]

        # Add position + frame embeddings
        flat = self.pos_embed(flat, S)

        # Enhanced Slot Attention
        slots, attn, all_attns = self.slot_attention(
            flat, view_mask=view_mask, temperature=temperature
        )

        return slots, attn, all_attns


# ─────────────────────────────────────────────
# 3D Box Head: regress 8 corners directly
# ─────────────────────────────────────────────

class CornersBoxHead(nn.Module):
    """Per-slot MLP → 8 corner points in normalized coordinates."""

    def __init__(self, dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 24),  # 8 corners × 3 coords
        )

    def forward(self, slots):
        """slots: [B, K, D] → [B, K, 8, 3]"""
        return self.head(slots).reshape(*slots.shape[:2], 8, 3)


class SlotClassifier(nn.Module):
    """Per-slot foreground/background classifier."""

    def __init__(self, dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2),  # [bg_logit, fg_logit]
        )

    def forward(self, slots):
        """slots: [B, K, D] → [B, K, 2]"""
        return self.head(slots)


# ─────────────────────────────────────────────
# GT Attention Mask Generation (Convex Hull)
# ─────────────────────────────────────────────

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]


def project_box_convex_hull_mask(box_corners_3d, K_intrinsic, pose,
                                 img_h, img_w, grid_h=37, grid_w=37,
                                 near_clip=0.1):
    """
    Project a single 3D box to 2D via near-plane clipping + convex hull fill.

    Handles:
      - Z < ε corners (near-plane clipping on 12 box edges)
      - Square padding offset from load_and_preprocess_images_square
      - Extreme projection coordinates (clip to [-2W, 2W])
      - Convex hull polygon clipped to image boundary
      - Minimum area filter (< 5 px² → discard)

    Args:
        box_corners_3d: [8, 3] world coordinates
        K_intrinsic: [3, 3]
        pose: [4, 4] world-to-camera
        img_h, img_w: original image dimensions
        grid_h, grid_w: patch grid size (37×37)
        near_clip: near plane Z threshold

    Returns:
        mask_grid: [grid_h, grid_w] float32 binary, or None if not visible
    """
    import cv2
    from scipy.spatial import ConvexHull

    # camera-to-world pose → world-to-camera: p_cam = R^T @ (p - t)
    corners_cam = world_to_camera(box_corners_3d, pose)  # [8, 3]

    # ── Check: all behind camera → skip ──
    if np.all(corners_cam[:, 2] <= near_clip):
        return None

    # ── Collect valid 3D points: front corners + edge-plane intersections ──
    valid_pts = []

    # Corners in front of near plane
    for i in range(8):
        if corners_cam[i, 2] > near_clip:
            valid_pts.append(corners_cam[i])

    # Clip 12 box edges against near plane
    for i, j in BOX_EDGES:
        zi, zj = corners_cam[i, 2], corners_cam[j, 2]
        if (zi > near_clip) != (zj > near_clip):
            t_val = (near_clip - zi) / (zj - zi + 1e-12)
            intersection = corners_cam[i] + t_val * (corners_cam[j] - corners_cam[i])
            valid_pts.append(intersection)

    if len(valid_pts) < 3:
        return None

    valid_pts = np.array(valid_pts)  # [N, 3]

    # ── Project to 2D (original image coords) ──
    fx, fy = K_intrinsic[0, 0], K_intrinsic[1, 1]
    cx, cy = K_intrinsic[0, 2], K_intrinsic[1, 2]
    z = valid_pts[:, 2]
    u = fx * valid_pts[:, 0] / z + cx
    v = fy * valid_pts[:, 1] / z + cy

    # Clip extreme coordinates
    max_bound = 2.0 * max(img_h, img_w)
    u = np.clip(u, -max_bound, max_bound)
    v = np.clip(v, -max_bound, max_bound)

    points_2d = np.stack([u, v], axis=1)  # [N, 2]

    # ── Convex hull ──
    if len(points_2d) < 3:
        return None

    try:
        hull = ConvexHull(points_2d)
        hull_pts = points_2d[hull.vertices]  # [H, 2]
    except Exception:
        return None

    # ── Transform to 518×518 coordinate space (square padding) ──
    max_dim = max(img_h, img_w)
    if img_h >= img_w:
        pad_x = (max_dim - img_w) / 2.0
        pad_y = 0.0
    else:
        pad_x = 0.0
        pad_y = (max_dim - img_h) / 2.0

    scale_518 = 518.0 / max_dim
    hull_518 = np.zeros_like(hull_pts)
    hull_518[:, 0] = (hull_pts[:, 0] + pad_x) * scale_518
    hull_518[:, 1] = (hull_pts[:, 1] + pad_y) * scale_518

    # Clip to image boundary
    hull_518[:, 0] = np.clip(hull_518[:, 0], 0, 517)
    hull_518[:, 1] = np.clip(hull_518[:, 1], 0, 517)
    hull_int = hull_518.astype(np.int32)

    # ── Minimum area filter ──
    area = cv2.contourArea(hull_int)
    if area < 5:
        return None

    # ── Fill convex hull on 518×518 canvas ──
    mask_518 = np.zeros((518, 518), dtype=np.uint8)
    cv2.fillPoly(mask_518, [hull_int], 1)

    # ── Downsample to patch grid (37×37) ──
    patch_size = 518 // grid_h  # 14
    mask_grid = mask_518.reshape(grid_h, patch_size, grid_w, patch_size)
    mask_grid = mask_grid.mean(axis=(1, 3))
    mask_grid = (mask_grid > 0.3).astype(np.float32)

    return mask_grid


def generate_gt_attention_masks(bbox_corners, K_intrinsic, all_poses,
                                selected_indices, img_h, img_w,
                                grid_h=37, grid_w=37):
    """
    Project 3D boxes to each selected view via convex hull fill.

    Returns:
        masks: [N_gt, S*n_patches] float32 (binary 0/1)
        visible: [N_gt] bool — whether instance has any visible token
    """
    S = len(selected_indices)
    N_gt = len(bbox_corners)
    n_patches = grid_h * grid_w

    masks = np.zeros((N_gt, S * n_patches), dtype=np.float32)

    for view_idx, cam_idx in enumerate(selected_indices):
        pose = all_poses[cam_idx]
        offset = view_idx * n_patches

        for gt_idx in range(N_gt):
            mask_grid = project_box_convex_hull_mask(
                bbox_corners[gt_idx], K_intrinsic, pose,
                img_h, img_w, grid_h, grid_w,
            )
            if mask_grid is not None:
                masks[gt_idx, offset:offset + n_patches] = mask_grid.flatten()

    visible = masks.sum(axis=1) > 0
    return masks, visible


# ─────────────────────────────────────────────
# Normalize GT Corners
# ─────────────────────────────────────────────

def normalize_gt_corners(bbox_corners, first_pose):
    """
    Transform GT corners to first-camera-centered normalized coordinates.

    Args:
        bbox_corners: [N_gt, 8, 3] world coordinates
        first_pose: [4, 4] first camera's world-to-camera transform

    Returns:
        corners_norm: [N_gt, 8, 3] normalized
        center: [3] scene center
        scale: float normalization scale
    """
    # camera-to-world pose: world-to-camera = R^T @ (p - t)
    R = first_pose[:3, :3]
    t = first_pose[:3, 3]
    shifted = bbox_corners - t
    corners_cam = np.einsum("bpi,ij->bpj", shifted, R)

    all_pts = corners_cam.reshape(-1, 3)
    center = all_pts.mean(axis=0)
    dists = np.linalg.norm(all_pts - center, axis=1)
    scale = max(dists.mean(), 1e-6)

    corners_norm = (corners_cam - center) / scale
    return corners_norm.astype(np.float32), center, scale


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────

def chamfer_loss(pred_corners, gt_corners):
    """
    Chamfer distance between predicted and GT 8-corner boxes.
    pred_corners: [N, 8, 3], gt_corners: [N, 8, 3]
    """
    dist = torch.cdist(pred_corners, gt_corners)  # [N, 8, 8]
    return dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()


def compute_box_matching_cost(pred_corners, pred_logits, gt_corners):
    """
    Cost matrix for Hungarian matching.
    pred_corners: [K, 8, 3], pred_logits: [K, 2], gt_corners: [M, 8, 3]
    Returns: cost [K, M] numpy
    """
    K, M = pred_corners.shape[0], gt_corners.shape[0]

    pred_exp = pred_corners.unsqueeze(1).expand(K, M, 8, 3).reshape(K * M, 8, 3)
    gt_exp = gt_corners.unsqueeze(0).expand(K, M, 8, 3).reshape(K * M, 8, 3)
    dist = torch.cdist(pred_exp, gt_exp)  # [K*M, 8, 8]
    cham = (dist.min(dim=2)[0].mean(dim=1) + dist.min(dim=1)[0].mean(dim=1))
    cost_chamfer = cham.reshape(K, M)

    cls_cost = -F.log_softmax(pred_logits, dim=-1)[:, 1:2].expand(K, M)
    cost = cost_chamfer + 2.0 * cls_cost
    return cost.detach().cpu().numpy()


def compute_mask_matching_cost(attn, gt_masks, pred_logits):
    """
    Mask-based cost matrix for Hungarian matching (Mask2Former style).
    Uses Dice cost + classification cost. No box cost.
    attn:        [K, N] attention maps (post-softmax probabilities)
    gt_masks:    [M, N] binary GT masks
    pred_logits: [K, 2] foreground/background logits
    Returns: cost [K, M] numpy
    """
    K, N = attn.shape
    M = gt_masks.shape[0]

    # Pairwise Dice cost: 1 - 2*intersection / union for each (slot, gt) pair
    # attn: [K, 1, N] vs gt_masks: [1, M, N]
    pred_exp = attn.unsqueeze(1)       # [K, 1, N]
    gt_exp = gt_masks.unsqueeze(0)     # [1, M, N]
    intersection = (pred_exp * gt_exp).sum(dim=-1)     # [K, M]
    union = pred_exp.sum(dim=-1) + gt_exp.sum(dim=-1)  # [K, M]
    dice_cost = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)  # [K, M]

    # Classification cost: -log(p_fg) for each slot, expanded to [K, M]
    cls_cost = -F.log_softmax(pred_logits, dim=-1)[:, 1:2].expand(K, M)

    cost = dice_cost + 1.0 * cls_cost
    return cost.detach().cpu().numpy()


def hungarian_matching(cost_matrix):
    """Solve assignment. Returns (row_ind, col_ind)."""
    from scipy.optimize import linear_sum_assignment
    return linear_sum_assignment(cost_matrix)


def prob_focal_loss(pred_prob, gt_mask, alpha=0.25, gamma=2.0, eps=1e-6):
    """
    Focal Loss for probability inputs (post-softmax attention maps).
    pred_prob: [P, N] attention probabilities
    gt_mask:   [P, N] binary GT mask (0/1)
    """
    p = torch.clamp(pred_prob, min=eps, max=1.0 - eps)

    # Foreground loss: -alpha * (1-p)^gamma * log(p) * gt
    pos_loss = -alpha * torch.pow(1 - p, gamma) * torch.log(p) * gt_mask

    # Background loss: -(1-alpha) * p^gamma * log(1-p) * (1-gt)
    neg_loss = -(1 - alpha) * torch.pow(p, gamma) * torch.log(1 - p) * (1 - gt_mask)

    return (pos_loss + neg_loss).mean()


def attention_mask_loss(attn, gt_masks, matched_slots, matched_instances):
    """
    L_mask = 5.0 * Dice (matched only) + 5.0 * Focal (all K slots).

    Global Focal: constructs full [K, N] target where matched slots get GT mask,
    unmatched slots get all-zeros. Pushes unmatched slots toward zero attention.
    Matched Dice: only on matched pairs (Dice is undefined for all-zero targets).

    attn: [K, N], gt_masks: [M, N] (float 0/1)
    """
    K, N = attn.shape
    device = attn.device

    # ── Global Focal Loss (all K slots) ──
    # Build full target: matched slots → GT mask, unmatched → zeros
    full_target = torch.zeros(K, N, device=device)
    if len(matched_slots) > 0:
        full_target[matched_slots] = gt_masks[matched_instances]
    focal_loss = prob_focal_loss(attn, full_target)

    # ── Matched Dice Loss (only matched pairs) ──
    if len(matched_slots) == 0:
        dice_loss = torch.tensor(0.0, device=device)
    else:
        pred = attn[matched_slots]
        target = gt_masks[matched_instances]
        intersection = (pred * target).sum(dim=-1)
        union = pred.sum(dim=-1) + target.sum(dim=-1)
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice.mean()

    return 5.0 * dice_loss + 5.0 * focal_loss


def hardneg_classification_loss(pred_logits, matched_slot_indices, num_hard_neg=5):
    """
    Hard Negative Mining: matched slots=positive, top-5 hardest unmatched=negative.
    pred_logits: [K, 2]
    """
    K = pred_logits.shape[0]
    device = pred_logits.device
    pos_set = set(int(x) for x in matched_slot_indices)
    neg_indices = [i for i in range(K) if i not in pos_set]

    if len(pos_set) == 0:
        return torch.tensor(0.0, device=device)

    pos_idx = torch.tensor(list(pos_set), device=device, dtype=torch.long)
    pos_loss = F.cross_entropy(
        pred_logits[pos_idx],
        torch.ones(len(pos_set), device=device, dtype=torch.long),
    )

    if len(neg_indices) == 0:
        return pos_loss

    neg_idx = torch.tensor(neg_indices, device=device, dtype=torch.long)
    neg_logits = pred_logits[neg_idx]
    neg_fg_prob = F.softmax(neg_logits, dim=-1)[:, 1]

    n_hard = min(num_hard_neg * len(pos_set), len(neg_indices))
    _, hard_sel = neg_fg_prob.topk(n_hard)

    neg_loss = F.cross_entropy(
        neg_logits[hard_sel],
        torch.zeros(n_hard, device=device, dtype=torch.long),
    )

    return pos_loss + neg_loss
