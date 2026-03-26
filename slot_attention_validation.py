"""
Step 1 Validation: Enhanced Slot Attention on VGGT Features

Validates that ESA can separate different objects into different slots
using only token-level objectness supervision (no 3D boxes, no contrastive).

Pipeline:
  1. Extract VGGT patch tokens (last layer) for CA1M scenes → cache in RAM
  2. Train ESA with token-level objectness BCE loss
  3. Visualize per-slot attention maps overlaid on original images

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n superbox python slot_attention_validation.py
"""

import os
import sys
import glob
import logging
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Setup paths
VGGT_ROOT = "/home/fanwg/myproject/code/vggt"
sys.path.insert(0, VGGT_ROOT)
sys.path.insert(0, os.path.join(VGGT_ROOT, "training"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CA1M_DIR = "/data1/lyq/CA1M-dataset/CA1M-dataset/training"
MODEL_CKPT = "/home/lanyuqing/model/model.pt"

IMG_SIZE = 518
PATCH_SIZE = 14
GRID_H = IMG_SIZE // PATCH_SIZE  # 37
GRID_W = IMG_SIZE // PATCH_SIZE  # 37
N_PATCHES = GRID_H * GRID_W  # 1369

# Training config
NUM_TRAIN_SCENES = 180
NUM_VAL_SCENES = 20
IMGS_PER_SCENE = 4
NUM_EPOCHS = 60
LR = 3e-4
WEIGHT_DECAY = 1e-4

# ESA config
SLOT_DIM = 256
NUM_SLOTS = 32
NUM_GLOBAL_SLOTS = 2
NUM_ITERATIONS = 3

VIS_DIR = os.path.join(VGGT_ROOT, "vis_slot_attention")


# ─────────────────────────────────────────────
# Data Loading (reused from linear_probe.py)
# ─────────────────────────────────────────────

def load_scene_images_by_indices(scene_dir, indices):
    """Load scene RGB images for specific frame indices."""
    from vggt.utils.load_fn import load_and_preprocess_images_square

    rgb_dir = os.path.join(scene_dir, "rgb")
    img_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    if not img_files:
        img_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    if len(img_files) < 2:
        return None, None

    selected = [img_files[i] for i in indices if i < len(img_files)]
    if len(selected) < 2:
        return None, None

    images, _ = load_and_preprocess_images_square(selected, IMG_SIZE)
    return images, selected


def load_scene_annotations(scene_dir):
    """Load camera params and 3D bounding boxes."""
    K_file = os.path.join(scene_dir, "K_rgb.txt")
    if not os.path.exists(K_file):
        return None
    K = np.loadtxt(K_file).reshape(3, 3).astype(np.float32)

    poses_file = os.path.join(scene_dir, "all_poses.npy")
    if not os.path.exists(poses_file):
        return None
    all_poses = np.load(poses_file).astype(np.float32)

    corners_file = os.path.join(scene_dir, "after_filter_boxes.npy")
    if not os.path.exists(corners_file):
        return None
    bbox_corners = np.load(corners_file).astype(np.float32)

    return {"K": K, "all_poses": all_poses, "bbox_corners": bbox_corners}


# ─────────────────────────────────────────────
# Feature Extraction & Caching
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_and_cache_features(model, scene_dirs, max_scenes, imgs_per_scene=4):
    """
    Extract VGGT patch tokens and objectness labels using CO-VISIBILITY camera selection.

    For each scene:
      1. Compute per-camera box visibility matrix (vectorized)
      2. Greedy-select cameras maximizing co-visibility (frustum overlap)
      3. Run VGGT on selected images
      4. Generate per-view instance labels from GT projection

    Returns list of dicts with features, labels, paths.
    """
    from enhanced_slot_attention import (
        project_boxes_to_objectness,
        project_boxes_to_instance_labels,
        select_covisible_cameras,
    )

    cache = []
    success = 0
    skipped_no_covis = 0

    for i, scene_dir in enumerate(scene_dirs):
        if success >= max_scenes:
            break

        anno = load_scene_annotations(scene_dir)
        if anno is None:
            continue

        K_mat = anno["K"]
        all_poses = anno["all_poses"]
        bbox_corners = anno["bbox_corners"]
        orig_h = int(2 * K_mat[1, 2])
        orig_w = int(2 * K_mat[0, 2])

        if len(bbox_corners) == 0 or len(all_poses) < imgs_per_scene:
            continue

        # ── Co-visibility camera selection ──
        sel_indices = select_covisible_cameras(
            bbox_corners, all_poses, K_mat, orig_h, orig_w,
            num_cameras=imgs_per_scene,
        )
        if sel_indices is None:
            skipped_no_covis += 1
            if skipped_no_covis % 50 == 0:
                log.info(f"  (skipped {skipped_no_covis} scenes: insufficient co-visibility)")
            continue

        # ── Load selected images ──
        images, img_paths = load_scene_images_by_indices(scene_dir, sel_indices)
        if images is None:
            continue

        # ── VGGT forward ──
        images_gpu = images.to("cuda")
        if images_gpu.dim() == 4:
            images_gpu = images_gpu.unsqueeze(0)

        try:
            agg_list, patch_start = model.aggregator(images_gpu)
        except Exception as e:
            log.warning(f"Scene {i} failed: {e}")
            del images_gpu
            torch.cuda.empty_cache()
            continue

        last = agg_list[-1]  # [1, S, P, 2048]
        S = last.shape[1]
        patch_tokens = last[:, :, patch_start:, :].squeeze(0)  # [S, 1369, 2048]
        features_cpu = patch_tokens.cpu().half()

        # ── Per-view objectness and instance labels ──
        obj_labels = np.zeros((S, N_PATCHES), dtype=np.float32)
        inst_labels = np.zeros((S, N_PATCHES), dtype=np.int64)

        for frame_idx in range(min(S, len(sel_indices))):
            sel_idx = sel_indices[frame_idx]
            if sel_idx >= len(all_poses):
                continue
            pose = all_poses[sel_idx]
            obj_labels[frame_idx] = project_boxes_to_objectness(
                bbox_corners, K_mat, pose, orig_h, orig_w, GRID_H, GRID_W
            )
            inst_labels[frame_idx] = project_boxes_to_instance_labels(
                bbox_corners, K_mat, pose, orig_h, orig_w, GRID_H, GRID_W
            )

        pos_ratio = obj_labels.mean()
        if pos_ratio < 0.001:
            skipped_no_covis += 1
            del images_gpu, agg_list, last, patch_tokens, features_cpu
            torch.cuda.empty_cache()
            continue

        # Compute co-visibility stats
        n_obj_per_view = (obj_labels.sum(axis=1) > 0).astype(int)  # per-view has objects
        n_covis_views = n_obj_per_view.sum()

        cache.append({
            "features": features_cpu,
            "objectness": torch.from_numpy(obj_labels),
            "instance_ids": torch.from_numpy(inst_labels),
            "img_paths": img_paths,
            "scene_dir": scene_dir,
            "S": S,
            "n_covis_views": int(n_covis_views),
        })
        success += 1

        if success % 20 == 0:
            log.info(f"  [{success}/{max_scenes}] {Path(scene_dir).name} "
                     f"S={S} pos={pos_ratio:.3f} covis_views={n_covis_views}/{S}")

        del images_gpu, agg_list, last, patch_tokens
        torch.cuda.empty_cache()

    # Summary stats
    if cache:
        avg_pos = np.mean([c["objectness"].numpy().mean() for c in cache])
        avg_covis = np.mean([c["n_covis_views"] for c in cache])
        log.info(f"Cached {len(cache)} scenes (skipped {skipped_no_covis}), "
                 f"avg_pos={avg_pos:.3f}, avg_covis_views={avg_covis:.1f}/{imgs_per_scene}, "
                 f"memory ~{sum(c['features'].nelement()*2 for c in cache)/1e9:.2f} GB")
    return cache


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_esa(train_cache, val_cache, epochs, lr, device="cuda"):
    """Train ESA with token-level objectness loss."""
    from enhanced_slot_attention import ESAValidationModel

    esa = ESAValidationModel(
        in_dim=2048,
        slot_dim=SLOT_DIM,
        num_slots=NUM_SLOTS,
        num_global_slots=NUM_GLOBAL_SLOTS,
        num_iterations=NUM_ITERATIONS,
        max_views=IMGS_PER_SCENE + 4,
    ).to(device)

    optimizer = torch.optim.AdamW(esa.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    # Compute global pos_weight for balanced BCE
    all_obj = torch.cat([c["objectness"].flatten() for c in train_cache])
    pos_count = all_obj.sum().item()
    neg_count = len(all_obj) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], device=device)
    log.info(f"pos_weight={pos_weight.item():.2f} "
             f"(pos={pos_count:.0f}, neg={neg_count:.0f})")

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        esa.train()
        perm = np.random.permutation(len(train_cache))
        epoch_loss = 0.0
        n_steps = 0
        t0 = time.time()

        for idx in perm:
            scene = train_cache[idx]
            S = scene["S"]
            feat = scene["features"].float().unsqueeze(0).to(device)  # [1, S, 1369, 2048]
            gt_obj = scene["objectness"].float().to(device).reshape(1, -1)  # [1, S*1369]

            optimizer.zero_grad()

            with autocast():
                token_obj, slot_obj, slots, attn, all_attns = esa(feat, S)

            # BCE outside autocast (token_obj is already probability from competitive softmax)
            token_obj_f = token_obj.float().clamp(1e-6, 1 - 1e-6)
            weight = torch.where(gt_obj > 0.5, pos_weight, torch.ones_like(gt_obj))
            loss = F.binary_cross_entropy(token_obj_f, gt_obj, weight=weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(esa.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_steps += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - t0

        # Validation
        val_loss = evaluate_esa(esa, val_cache, device, pos_weight)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in esa.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
                f"best_val={best_val_loss:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"time={elapsed:.1f}s"
            )

    # Load best
    if best_state is not None:
        esa.load_state_dict(best_state)
    log.info(f"Training done. Best val_loss={best_val_loss:.4f}")

    return esa


@torch.no_grad()
def evaluate_esa(esa, cache, device, pos_weight):
    """Evaluate on validation set."""
    esa.eval()
    total_loss = 0.0
    n = 0

    for scene in cache:
        S = scene["S"]
        feat = scene["features"].float().unsqueeze(0).to(device)
        gt_obj = scene["objectness"].float().to(device).reshape(1, -1)

        with autocast():
            token_obj, _, _, _, _ = esa(feat, S)

        token_obj_f = token_obj.float().clamp(1e-6, 1 - 1e-6)
        with torch.no_grad():
            loss = F.binary_cross_entropy(
                token_obj_f, gt_obj,
                weight=torch.where(gt_obj > 0.5, pos_weight, torch.ones_like(gt_obj)),
            )

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

@torch.no_grad()
def visualize_slot_attention(esa, val_cache, save_dir, device="cuda",
                             num_scenes=8, top_k_slots=6):
    """
    For each scene, visualize:
      Row 0: Original images
      Rows 1..top_k: Top-K active slots' attention maps overlaid on images
      Last row: GT objectness overlay
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    os.makedirs(save_dir, exist_ok=True)
    esa.eval()

    # Color palette for slots
    slot_colors = plt.cm.tab20(np.linspace(0, 1, NUM_SLOTS))

    for scene_idx, scene in enumerate(val_cache[:num_scenes]):
        S = scene["S"]
        feat = scene["features"].float().unsqueeze(0).to(device)
        gt_obj = scene["objectness"].numpy()  # [S, 1369]
        img_paths = scene["img_paths"]

        token_obj, slot_obj, slots, attn, all_attns = esa(feat, S)

        # attn: [1, K, S*1369] → [K, S, 37, 37]
        attn_np = attn.squeeze(0).cpu().numpy()  # [K, S*1369]
        attn_per_view = attn_np.reshape(NUM_SLOTS, S, GRID_H, GRID_W)

        # Slot activity (mass)
        slot_mass = attn_np.sum(axis=-1)  # [K]
        # Skip global slots, rank by mass
        obj_slot_indices = np.argsort(-slot_mass[NUM_GLOBAL_SLOTS:]) + NUM_GLOBAL_SLOTS
        top_slots = obj_slot_indices[:top_k_slots]

        # Load original images for display
        from PIL import Image
        orig_images = []
        for p in img_paths[:S]:
            img = Image.open(p).convert("RGB")
            img = img.resize((GRID_W * 8, GRID_H * 8))  # upscale for display
            orig_images.append(np.array(img))

        # Create figure: (1 + top_k + 1) rows × S columns
        n_rows = 1 + top_k_slots + 1
        fig, axes = plt.subplots(n_rows, S, figsize=(4 * S, 3 * n_rows))
        if S == 1:
            axes = axes[:, np.newaxis]

        scene_name = Path(scene["scene_dir"]).name

        # Row 0: Original images
        for v in range(S):
            axes[0, v].imshow(orig_images[v])
            axes[0, v].set_title(f"View {v}", fontsize=9)
            axes[0, v].axis("off")

        # Rows 1..top_k: Slot attention overlays
        for rank, slot_idx in enumerate(top_slots):
            for v in range(S):
                axes[rank + 1, v].imshow(orig_images[v], alpha=0.4)

                heatmap = attn_per_view[slot_idx, v]  # [37, 37]
                # Upsample to image size
                heatmap_up = np.kron(
                    heatmap,
                    np.ones((8, 8))
                )
                axes[rank + 1, v].imshow(
                    heatmap_up,
                    cmap="hot",
                    alpha=0.6,
                    vmin=0,
                    vmax=max(heatmap.max(), 1e-6),
                )
                if v == 0:
                    axes[rank + 1, v].set_ylabel(
                        f"Slot {slot_idx}\nmass={slot_mass[slot_idx]:.3f}",
                        fontsize=8,
                    )
                axes[rank + 1, v].axis("off")

        # Last row: GT objectness
        for v in range(S):
            axes[-1, v].imshow(orig_images[v], alpha=0.4)
            gt_map = gt_obj[v].reshape(GRID_H, GRID_W)
            gt_up = np.kron(gt_map, np.ones((8, 8)))
            axes[-1, v].imshow(gt_up, cmap="Greens", alpha=0.6, vmin=0, vmax=1)
            if v == 0:
                axes[-1, v].set_ylabel("GT Object", fontsize=8)
            axes[-1, v].axis("off")

        fig.suptitle(f"Scene: {scene_name}", fontsize=12, y=1.0)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"scene_{scene_idx:03d}_{scene_name}.png")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved {save_path}")

    # ── Summary: Slot activity distribution ──
    log.info("Generating slot activity summary...")
    all_masses = []
    for scene in val_cache:
        S = scene["S"]
        feat = scene["features"].float().unsqueeze(0).to(device)
        _, _, _, attn, _ = esa(feat, S)
        mass = attn.squeeze(0).sum(dim=-1).cpu().numpy()  # [K]
        all_masses.append(mass)

    all_masses = np.stack(all_masses)  # [N_scenes, K]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Slot mass distribution
    mean_mass = all_masses.mean(axis=0)
    std_mass = all_masses.std(axis=0)
    x = np.arange(NUM_SLOTS)
    axes[0].bar(x, mean_mass, yerr=std_mass, color="steelblue", alpha=0.8)
    axes[0].axvspan(-0.5, NUM_GLOBAL_SLOTS - 0.5, alpha=0.15, color="gray",
                     label="Global slots")
    axes[0].set_xlabel("Slot Index")
    axes[0].set_ylabel("Attention Mass (mean ± std)")
    axes[0].set_title("Slot Activity Distribution")
    axes[0].legend()

    # Iteration-wise attention entropy
    # Re-run one scene to get all_attns
    scene = val_cache[0]
    feat = scene["features"].float().unsqueeze(0).to(device)
    _, _, _, _, all_attns = esa(feat, scene["S"])

    entropies = []
    for t, a in enumerate(all_attns):
        a_np = a.squeeze(0).cpu().numpy()  # [K, N]
        # Per-token entropy over slots
        a_safe = np.clip(a_np, 1e-10, 1.0)
        entropy = -(a_safe * np.log(a_safe)).sum(axis=0).mean()
        entropies.append(entropy)

    axes[1].plot(range(1, len(entropies) + 1), entropies, "o-", color="coral")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean Token Entropy (over slots)")
    axes[1].set_title("Attention Sharpening Over Iterations")
    axes[1].set_xticks(range(1, len(entropies) + 1))

    plt.tight_layout()
    summary_path = os.path.join(save_dir, "summary_stats.png")
    fig.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {summary_path}")

    # ── Combined slot segmentation map ──
    log.info("Generating combined slot segmentation maps...")
    for scene_idx, scene in enumerate(val_cache[:num_scenes]):
        S = scene["S"]
        feat = scene["features"].float().unsqueeze(0).to(device)
        _, _, _, attn, _ = esa(feat, S)
        attn_np = attn.squeeze(0).cpu().numpy()  # [K, S*1369]

        # Argmax assignment: each token → its dominant slot
        assignment = attn_np.argmax(axis=0).reshape(S, GRID_H, GRID_W)  # [S, 37, 37]

        img_paths = scene["img_paths"]
        from PIL import Image

        fig, axes = plt.subplots(2, S, figsize=(4 * S, 7))
        if S == 1:
            axes = axes[:, np.newaxis]

        for v in range(S):
            img = Image.open(img_paths[v]).convert("RGB").resize((GRID_W * 8, GRID_H * 8))
            axes[0, v].imshow(np.array(img))
            axes[0, v].set_title(f"View {v}", fontsize=9)
            axes[0, v].axis("off")

            seg = assignment[v]  # [37, 37]
            seg_up = np.kron(seg, np.ones((8, 8), dtype=int))
            seg_color = slot_colors[seg_up.flatten()].reshape(
                GRID_H * 8, GRID_W * 8, 4
            )
            axes[1, v].imshow(np.array(img), alpha=0.3)
            axes[1, v].imshow(seg_color, alpha=0.7)
            axes[1, v].axis("off")

        scene_name = Path(scene["scene_dir"]).name
        fig.suptitle(f"Slot Segmentation: {scene_name}", fontsize=11)
        plt.tight_layout()
        path = os.path.join(save_dir, f"segmap_{scene_idx:03d}_{scene_name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    log.info(f"All visualizations saved to {save_dir}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Step 1: Enhanced Slot Attention Validation")
    log.info("=" * 60)
    device = "cuda"

    import pickle
    cache_file = os.path.join(VGGT_ROOT, "esa_feature_cache.pkl")
    total = NUM_TRAIN_SCENES + NUM_VAL_SCENES

    if os.path.exists(cache_file):
        log.info(f"Loading cached features from {cache_file}...")
        with open(cache_file, "rb") as f:
            all_cache = pickle.load(f)
        log.info(f"Loaded {len(all_cache)} cached scenes")
    else:
        # 1. Load VGGT model (aggregator only)
        log.info("Loading VGGT model (aggregator only)...")
        from vggt.models.vggt import VGGT

        vggt = VGGT(
            enable_camera=False, enable_gravity=False, enable_depth=False,
            enable_point=False, enable_track=False, enable_cubify=False,
        )
        ckpt = torch.load(MODEL_CKPT, map_location="cpu")
        missing, unexpected = vggt.load_state_dict(ckpt, strict=False)
        log.info(f"VGGT loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        vggt = vggt.to(device).eval()

        # 2. Get scene list
        scene_dirs = sorted(glob.glob(os.path.join(CA1M_DIR, "*")))
        scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
        log.info(f"Found {len(scene_dirs)} CA1M scenes")

        np.random.seed(42)
        np.random.shuffle(scene_dirs)

        # 3. Extract and cache features
        log.info(f"\nExtracting features for {total} scenes...")
        t0 = time.time()
        all_cache = extract_and_cache_features(
            vggt, scene_dirs, max_scenes=total, imgs_per_scene=IMGS_PER_SCENE
        )
        log.info(f"Extraction done in {time.time()-t0:.1f}s")

        # Save cache
        log.info(f"Saving feature cache to {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump(all_cache, f)
        log.info("Cache saved.")

        # Free VGGT GPU memory
        del vggt
        torch.cuda.empty_cache()
        log.info("VGGT freed from GPU")

    # Split train/val
    train_cache = all_cache[:NUM_TRAIN_SCENES]
    val_cache = all_cache[NUM_TRAIN_SCENES:total]
    log.info(f"Train: {len(train_cache)} scenes, Val: {len(val_cache)} scenes")

    # 4. Train ESA
    log.info(f"\nTraining ESA (K={NUM_SLOTS}, T={NUM_ITERATIONS}, D={SLOT_DIM})...")
    t0 = time.time()
    esa = train_esa(
        train_cache, val_cache,
        epochs=NUM_EPOCHS, lr=LR, device=device,
    )
    log.info(f"Training done in {time.time()-t0:.1f}s")

    # Save model
    save_path = os.path.join(VGGT_ROOT, "esa_step1_best.pt")
    torch.save(esa.state_dict(), save_path)
    log.info(f"Model saved to {save_path}")

    # 5. Visualize
    log.info(f"\nGenerating visualizations...")
    visualize_slot_attention(
        esa, val_cache, VIS_DIR, device=device,
        num_scenes=min(8, len(val_cache)), top_k_slots=6,
    )

    log.info("\n" + "=" * 60)
    log.info("Step 1 Validation Complete!")
    log.info(f"Visualizations: {VIS_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
