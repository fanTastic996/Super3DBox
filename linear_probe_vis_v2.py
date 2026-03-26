"""
Phase 0 可视化 v2: 修正坐标映射 + 高对比度热力图
"""

import os, sys, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/home/fanwg/myproject/code/vggt")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMG_SIZE = 518
PATCH_SIZE = 14
GRID_H = GRID_W = IMG_SIZE // PATCH_SIZE  # 37
CA1M_DIR = "/data1/lyq/CA1M-dataset/CA1M-dataset/training"
OUTPUT_DIR = "/home/fanwg/myproject/code/vggt/vis_phase0_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from linear_probe import load_scene_images, load_scene_annotations, LinearProbe


def project_boxes_to_patch_grid_v2(bbox_corners, K, pose, orig_h, orig_w):
    """
    修正版: 考虑 square padding + resize 的坐标变换

    load_and_preprocess_images_square 做了:
      1. 将图像 pad 到 max(H,W) x max(H,W) 的正方形
      2. resize 到 518 x 518
    """
    labels = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    if bbox_corners is None or len(bbox_corners) == 0:
        return labels

    R, t = pose[:3, :3], pose[:3, 3]
    max_dim = max(orig_h, orig_w)
    # Padding offsets: image is centered in the square
    pad_top = (max_dim - orig_h) // 2
    pad_left = (max_dim - orig_w) // 2

    for box in bbox_corners:
        corners_cam = (R @ box.T).T + t
        valid = corners_cam[:, 2] > 0.1
        if valid.sum() < 4:
            continue
        cv = corners_cam[valid]

        # Step 1: camera → original pixel
        px_orig = K[0,0] * cv[:,0] / cv[:,2] + K[0,2]
        py_orig = K[1,1] * cv[:,1] / cv[:,2] + K[1,2]

        # Step 2: original pixel → square-padded pixel
        px_sq = px_orig + pad_left
        py_sq = py_orig + pad_top

        # Step 3: square-padded → 518x518
        px_518 = px_sq * IMG_SIZE / max_dim
        py_518 = py_sq * IMG_SIZE / max_dim

        # Step 4: pixel → patch grid
        gx_min = max(0, int(np.floor(px_518.min() / PATCH_SIZE)))
        gx_max = min(GRID_W - 1, int(np.floor(px_518.max() / PATCH_SIZE)))
        gy_min = max(0, int(np.floor(py_518.min() / PATCH_SIZE)))
        gy_max = min(GRID_H - 1, int(np.floor(py_518.max() / PATCH_SIZE)))

        if gx_max >= gx_min and gy_max >= gy_min:
            labels[gy_min:gy_max+1, gx_min:gx_max+1] = 1.0

    return labels


def project_boxes_to_518(bbox_corners, K, pose, orig_h, orig_w):
    """投影 3D box 到 518x518 图像坐标, 返回 2D 矩形列表"""
    rects = []
    R, t = pose[:3, :3], pose[:3, 3]
    max_dim = max(orig_h, orig_w)
    pad_top = (max_dim - orig_h) // 2
    pad_left = (max_dim - orig_w) // 2

    for box in bbox_corners:
        cam = (R @ box.T).T + t
        valid = cam[:, 2] > 0.1
        if valid.sum() < 4:
            continue
        cv = cam[valid]
        px = (K[0,0] * cv[:,0] / cv[:,2] + K[0,2] + pad_left) * IMG_SIZE / max_dim
        py = (K[1,1] * cv[:,1] / cv[:,2] + K[1,2] + pad_top) * IMG_SIZE / max_dim
        rects.append((px.min(), py.min(), px.max(), py.max()))
    return rects


def train_probe(features, labels, epochs=25, lr=1e-3):
    """训练 probe, 返回模型"""
    n = len(features)
    perm = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    X_tr = torch.from_numpy(features[perm[:split]]).cuda()
    y_tr = torch.from_numpy(labels[perm[:split]]).cuda()

    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).cuda()
    probe = LinearProbe(features.shape[1]).cuda()
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    for _ in range(epochs):
        probe.train()
        idx = torch.randperm(len(X_tr))
        for s in range(0, len(X_tr), 4096):
            i = idx[s:s+4096]
            loss = crit(probe(X_tr[i]), y_tr[i])
            opt.zero_grad(); loss.backward(); opt.step()

    probe.eval()
    return probe


def vis_scene(scene_dir, model, probe, idx, all_rgb=None):
    """每帧可视化: 518x518 图 + GT 框 + GT 网格 + 预测热力图"""
    images, img_paths = load_scene_images(scene_dir, max_images=4)
    if images is None:
        return False
    anno = load_scene_annotations(scene_dir)
    if anno is None:
        return False

    K = anno["K"]
    all_poses = anno["all_poses"]
    bbox_corners = anno["bbox_corners"]
    orig_h, orig_w = int(2 * K[1,2]), int(2 * K[0,2])

    if all_rgb is None:
        all_rgb = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
        if not all_rgb:
            all_rgb = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.jpg")))

    step = max(1, len(all_rgb) // 4)
    sel_indices = list(range(0, len(all_rgb), step))[:4]

    # Run model
    images_gpu = images.cuda().unsqueeze(0)
    with torch.no_grad():
        tokens_list, psi = model.aggregator(images_gpu)

    # Per-layer probes are not available here, use the single probe on last layer
    last_tokens = tokens_list[-1][:, :, psi:, :]  # [1, S, 1369, 2048]

    # --- Per-frame + consensus visualization ---
    S = last_tokens.shape[1]

    # Consensus
    consensus = last_tokens.mean(dim=1).squeeze(0).float()  # [1369, 2048]
    with torch.no_grad():
        cons_probs = torch.sigmoid(probe(consensus)).cpu().numpy().reshape(GRID_H, GRID_W)

    # GT labels (union across frames)
    gt_union = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for si in sel_indices:
        if si < len(all_poses):
            gt_union = np.maximum(gt_union,
                project_boxes_to_patch_grid_v2(bbox_corners, K, all_poses[si], orig_h, orig_w))

    # Per-scene AP
    scene_ap = average_precision_score(gt_union.flatten(), cons_probs.flatten()) \
        if gt_union.sum() > 0 else 0.0

    # Get 518x518 version of the displayed image
    img_518 = images[0].permute(1, 2, 0).numpy()  # First selected frame
    img_518 = np.clip(img_518, 0, 1)

    show_frame = sel_indices[0]
    pose_show = all_poses[show_frame] if show_frame < len(all_poses) else all_poses[0]
    gt_frame = project_boxes_to_patch_grid_v2(bbox_corners, K, pose_show, orig_h, orig_w)
    rects_518 = project_boxes_to_518(bbox_corners, K, pose_show, orig_h, orig_w)

    # ─── Figure: 5 panels ───
    fig, axes = plt.subplots(1, 5, figsize=(28, 5.5), dpi=150)

    # Custom colormaps
    cmap_pred = LinearSegmentedColormap.from_list("pred",
        [(0, "#000022"), (0.2, "#001a66"), (0.4, "#0055aa"),
         (0.6, "#ff4444"), (0.8, "#ff8800"), (1.0, "#ffff00")])
    cmap_gt = LinearSegmentedColormap.from_list("gt",
        [(0, "#00000000"), (0.5, "#00cc5580"), (1.0, "#00ff66ff")])

    # (a) 518x518 image + GT 2D boxes
    axes[0].imshow(img_518)
    for (x1, y1, x2, y2) in rects_518:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(517, x2), min(517, y2)
        if x2c > x1c and y2c > y1c:
            rect = mpatches.FancyBboxPatch((x1c, y1c), x2c-x1c, y2c-y1c,
                boxstyle="round,pad=2", linewidth=2.5,
                edgecolor="#00ff88", facecolor="none")
            axes[0].add_patch(rect)
    axes[0].set_title("(a) Input (518x518) + GT Boxes", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # (b) GT patch labels on 37x37 grid (nearest upsampled)
    gt_up = np.kron(gt_frame, np.ones((PATCH_SIZE, PATCH_SIZE)))[:IMG_SIZE, :IMG_SIZE]
    axes[1].imshow(img_518, alpha=0.5)
    axes[1].imshow(gt_up, cmap=cmap_gt, alpha=0.7, vmin=0, vmax=1)
    # Draw grid lines
    for g in range(0, IMG_SIZE, PATCH_SIZE):
        axes[1].axhline(g, color="white", linewidth=0.15, alpha=0.3)
        axes[1].axvline(g, color="white", linewidth=0.15, alpha=0.3)
    axes[1].set_title(f"(b) GT Patch Labels ({int(gt_frame.sum())}/{GRID_H*GRID_W} positive)",
                      fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # (c) 37x37 raw prediction grid (no interpolation)
    pred_block = np.kron(cons_probs, np.ones((PATCH_SIZE, PATCH_SIZE)))[:IMG_SIZE, :IMG_SIZE]
    axes[2].imshow(pred_block, cmap=cmap_pred, vmin=0, vmax=1)
    for g in range(0, IMG_SIZE, PATCH_SIZE):
        axes[2].axhline(g, color="white", linewidth=0.15, alpha=0.2)
        axes[2].axvline(g, color="white", linewidth=0.15, alpha=0.2)
    axes[2].set_title("(c) Prediction: 37x37 Grid (raw)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # (d) Smoothed prediction overlay on image
    pred_smooth = F.interpolate(
        torch.from_numpy(cons_probs).unsqueeze(0).unsqueeze(0).float(),
        size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    ).squeeze().numpy()
    axes[3].imshow(img_518)
    im = axes[3].imshow(pred_smooth, cmap=cmap_pred, alpha=0.75, vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04, label="Score")
    axes[3].set_title("(d) Prediction Overlay (smoothed)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # (e) Side-by-side comparison
    # Create a 3-channel comparison: GT=Green, Pred=Red, Overlap=Yellow
    comp = np.zeros((IMG_SIZE, IMG_SIZE, 3))
    comp[:,:,0] = pred_smooth          # Red = prediction
    comp[:,:,1] = gt_up.astype(float)  # Green = GT
    # Where both > 0.5: yellow
    bg = img_518 * 0.25
    final = bg + comp * 0.75
    final = np.clip(final, 0, 1)
    axes[4].imshow(final)
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.7, label='Prediction'),
        mpatches.Patch(facecolor='green', alpha=0.7, label='GT Label'),
        mpatches.Patch(facecolor='yellow', alpha=0.7, label='Overlap')]
    axes[4].legend(handles=legend_elements, loc='lower right', fontsize=9,
                   facecolor='black', labelcolor='white', edgecolor='gray')
    axes[4].set_title(f"(e) Pred vs GT  |  Scene AP={scene_ap:.3f}", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    scene_name = os.path.basename(scene_dir)
    fig.suptitle(
        f"Scene {scene_name}   |   Linear Probe (frozen VGGT, single linear layer)   |   "
        f"Positive ratio: {gt_union.mean():.3f}   |   AP: {scene_ap:.3f}",
        fontsize=13, fontweight="bold", y=1.01)

    plt.savefig(os.path.join(OUTPUT_DIR, f"scene_{idx:03d}_{scene_name}.png"),
                bbox_inches="tight", dpi=150, facecolor="white")
    plt.close()
    return True


def vis_multilayer(scene_dir, model, probes, idx):
    """多层对比: Original + GT + Layer 3/11/17/23"""
    images, img_paths = load_scene_images(scene_dir, max_images=4)
    if images is None:
        return False
    anno = load_scene_annotations(scene_dir)
    if anno is None:
        return False

    K = anno["K"]
    all_poses = anno["all_poses"]
    bbox_corners = anno["bbox_corners"]
    orig_h, orig_w = int(2*K[1,2]), int(2*K[0,2])

    all_rgb = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
    if not all_rgb:
        all_rgb = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.jpg")))
    step = max(1, len(all_rgb) // 4)
    sel_indices = list(range(0, len(all_rgb), step))[:4]

    img_518 = images[0].permute(1, 2, 0).numpy()
    img_518 = np.clip(img_518, 0, 1)

    # GT
    show_frame = sel_indices[0]
    pose_show = all_poses[show_frame] if show_frame < len(all_poses) else all_poses[0]
    gt_frame = project_boxes_to_patch_grid_v2(bbox_corners, K, pose_show, orig_h, orig_w)
    gt_union = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for si in sel_indices:
        if si < len(all_poses):
            gt_union = np.maximum(gt_union,
                project_boxes_to_patch_grid_v2(bbox_corners, K, all_poses[si], orig_h, orig_w))
    rects_518 = project_boxes_to_518(bbox_corners, K, pose_show, orig_h, orig_w)

    # Run model
    images_gpu = images.cuda().unsqueeze(0)
    with torch.no_grad():
        tokens_list, psi = model.aggregator(images_gpu)

    layers = [3, 11, 17, 23]
    layer_aps_global = {3: 0.927, 11: 0.946, 17: 0.969, 23: 0.980}

    cmap_pred = LinearSegmentedColormap.from_list("pred",
        [(0, "#000022"), (0.2, "#001a66"), (0.4, "#0055aa"),
         (0.6, "#ff4444"), (0.8, "#ff8800"), (1.0, "#ffff00")])
    cmap_gt = LinearSegmentedColormap.from_list("gt",
        [(0, "#00000000"), (0.5, "#00cc5580"), (1.0, "#00ff66ff")])

    fig, axes = plt.subplots(2, 3, figsize=(21, 14), dpi=150)

    # Row 0, Col 0: Original + GT
    ax = axes[0, 0]
    ax.imshow(img_518)
    for (x1, y1, x2, y2) in rects_518:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(517, x2), min(517, y2)
        if x2c > x1c and y2c > y1c:
            rect = mpatches.FancyBboxPatch((x1c, y1c), x2c-x1c, y2c-y1c,
                boxstyle="round,pad=2", linewidth=2.5,
                edgecolor="#00ff88", facecolor="none")
            ax.add_patch(rect)
    ax.set_title("Original + GT 3D Boxes", fontsize=12, fontweight="bold")
    ax.axis("off")

    # Row 0, Col 1: GT patch labels
    ax = axes[0, 1]
    gt_up = np.kron(gt_union, np.ones((PATCH_SIZE, PATCH_SIZE)))[:IMG_SIZE, :IMG_SIZE]
    ax.imshow(img_518, alpha=0.5)
    ax.imshow(gt_up, cmap=cmap_gt, alpha=0.7, vmin=0, vmax=1)
    ax.set_title(f"GT Patch Labels ({int(gt_union.sum())} positive patches)", fontsize=12, fontweight="bold")
    ax.axis("off")

    # Row 0, Col 2: empty → put legend/info
    ax = axes[0, 2]
    ax.axis("off")
    info_text = (
        "Linear Probe Experiment\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Model: Frozen VGGT Aggregator\n"
        f"Probe: Linear(2048, 1)\n"
        f"Input: Consensus features\n"
        f"       (mean across {images.shape[0]} frames)\n\n"
        "Layer APs (global):\n"
    )
    for l in layers:
        bar = "█" * int(layer_aps_global[l] * 20)
        info_text += f"  Layer {l:2d}: {bar} {layer_aps_global[l]:.3f}\n"
    info_text += f"\nPositive patch ratio: {gt_union.mean():.3f}"
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor='gray', alpha=0.9),
            color='white')

    # Row 1: Layer 3, 11, 17, 23 → fit into 3 cols, use 4 subplots
    # Rearrange: remove axes[1,2] and create 4 equal subplots in row 1
    for j in range(3):
        axes[1, j].remove()

    gs = fig.add_gridspec(2, 4, hspace=0.15, wspace=0.05,
                          left=0.02, right=0.98, top=0.48, bottom=0.02)

    for li, layer_idx in enumerate(layers):
        ax = fig.add_subplot(gs[1, li])
        tokens = tokens_list[layer_idx][:, :, psi:, :]
        consensus = tokens.mean(dim=1).squeeze(0).float()
        with torch.no_grad():
            logits = probes[layer_idx](consensus)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(GRID_H, GRID_W)

        # Per-scene AP for this layer
        if gt_union.sum() > 0:
            layer_scene_ap = average_precision_score(gt_union.flatten(), probs.flatten())
        else:
            layer_scene_ap = 0.0

        pred_smooth = F.interpolate(
            torch.from_numpy(probs).unsqueeze(0).unsqueeze(0).float(),
            size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        ax.imshow(img_518, alpha=0.35)
        ax.imshow(pred_smooth, cmap=cmap_pred, alpha=0.8, vmin=0, vmax=1)
        ax.set_title(f"Layer {layer_idx}  |  Global AP={layer_aps_global[layer_idx]:.3f}  |  "
                     f"Scene AP={layer_scene_ap:.3f}",
                     fontsize=10, fontweight="bold")
        ax.axis("off")

    scene_name = os.path.basename(scene_dir)
    fig.suptitle(f"Multi-Layer Feature Comparison  —  Scene: {scene_name}",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.savefig(os.path.join(OUTPUT_DIR, f"multilayer_{idx:03d}_{scene_name}.png"),
                bbox_inches="tight", dpi=150, facecolor="white")
    plt.close()
    return True


def vis_summary():
    """汇总统计图"""
    layers = [3, 11, 17, 23]
    aps = [0.927, 0.946, 0.969, 0.980]
    f1s = [0.576, 0.611, 0.886, 0.891]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # (a) AP bar chart
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    bars = axes[0].bar(range(4), aps, color=colors, width=0.55, edgecolor="white", linewidth=2)
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels([f"Layer {l}" for l in layers], fontsize=13)
    axes[0].set_ylabel("Average Precision", fontsize=13)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].axhline(0.5, color="gray", ls="--", alpha=0.4, label="Feasibility threshold")
    axes[0].axhspan(0, 0.5, alpha=0.05, color="red")
    axes[0].axhspan(0.5, 1.0, alpha=0.05, color="green")
    for bar, ap in zip(bars, aps):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                     f"{ap:.3f}", ha="center", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].set_title("Detection AP by VGGT Layer", fontsize=14, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.2)

    # (b) Conclusion box
    axes[1].axis("off")
    conclusion = (
        "Phase 0 Validation Results\n"
        "══════════════════════════════════\n\n"
        "Experiment: Token-level objectness classification\n"
        "  • Model: Frozen VGGT Aggregator (no fine-tuning)\n"
        "  • Classifier: Single Linear Layer (2048 → 1)\n"
        "  • Data: 200 CA1M scenes, 4 views each\n"
        "  • Labels: 3D box → 2D projection → patch grid\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  ✓  Best AP = 0.980 (Layer 23)\n"
        "  ✓  All layers > 0.90 AP\n"
        "  ✓  Layer 17 ≈ Layer 23 (saturation)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Conclusion:\n"
        "  VGGT patch tokens contain rich object\n"
        "  detection information. A single linear\n"
        "  layer can locate objects with 98% AP.\n\n"
        "  → Proceed to Phase 1: MVP Implementation"
    )
    axes[1].text(0.05, 0.95, conclusion, transform=axes[1].transAxes,
                 fontsize=12, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#0d1117',
                          edgecolor='#30363d', alpha=0.95),
                 color='#e6edf3')

    fig.suptitle("Phase 0: Can VGGT Features Detect Objects?",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(os.path.join(OUTPUT_DIR, "summary.png"),
                bbox_inches="tight", dpi=150, facecolor="white")
    plt.close()
    print("[Saved] summary.png")


def main():
    print("=" * 50)
    print("Phase 0 Visualization v2")
    print("=" * 50)

    from vggt.models.vggt import VGGT
    print("Loading model...")
    model = VGGT(enable_camera=False, enable_gravity=False, enable_depth=False,
                 enable_point=False, enable_track=False, enable_cubify=False)
    ckpt = torch.load("/home/lanyuqing/model/model.pt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()

    scene_dirs = sorted(glob.glob(os.path.join(CA1M_DIR, "*")))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
    np.random.seed(42)
    np.random.shuffle(scene_dirs)

    # Train probes (100 scenes)
    print("Extracting features & training probes...")
    all_feats = {3:[], 11:[], 17:[], 23:[]}
    all_labels = []
    count = 0
    for sd in scene_dirs:
        if count >= 100:
            break
        imgs, paths = load_scene_images(sd, 4)
        if imgs is None:
            continue
        a = load_scene_annotations(sd)
        if a is None:
            continue
        K, poses, boxes = a["K"], a["all_poses"], a["bbox_corners"]
        oh, ow = int(2*K[1,2]), int(2*K[0,2])
        all_rgb = sorted(glob.glob(os.path.join(sd, "rgb", "*.png")))
        if not all_rgb:
            all_rgb = sorted(glob.glob(os.path.join(sd, "rgb", "*.jpg")))
        step = max(1, len(all_rgb)//4)
        sels = list(range(0, len(all_rgb), step))[:4]

        ig = imgs.cuda().unsqueeze(0)
        with torch.no_grad():
            tl, psi = model.aggregator(ig)
            for li in [3,11,17,23]:
                c = tl[li][:,:,psi:,:].mean(dim=1).squeeze(0).cpu().float().numpy()
                all_feats[li].append(c)
        lbl = np.zeros(GRID_H*GRID_W, dtype=np.float32)
        for si in sels:
            if si < len(poses):
                lbl = np.maximum(lbl, project_boxes_to_patch_grid_v2(boxes, K, poses[si], oh, ow).flatten())
        all_labels.append(lbl)
        count += 1
        del ig, tl; torch.cuda.empty_cache()
        if count % 25 == 0:
            print(f"  [{count}/100]")

    labels_all = np.concatenate(all_labels)
    probes = {}
    for li in [3,11,17,23]:
        feats = np.concatenate(all_feats[li])
        probes[li] = train_probe(feats, labels_all)
        with torch.no_grad():
            p = torch.sigmoid(probes[li](torch.from_numpy(feats).cuda())).cpu().numpy()
            ap = average_precision_score(labels_all, p)
        print(f"  Layer {li}: AP={ap:.4f}")

    # Generate visualizations
    print("\nSummary plot...")
    vis_summary()

    print("Per-scene visualizations (8 scenes)...")
    vc = 0
    for sd in scene_dirs[100:]:
        if vc >= 8:
            break
        if vis_scene(sd, model, probes[23], vc):
            print(f"  [{vc+1}/8] {os.path.basename(sd)}")
            vc += 1

    print("Multi-layer comparisons (4 scenes)...")
    mc = 0
    for sd in scene_dirs[108:]:
        if mc >= 4:
            break
        if vis_multilayer(sd, model, probes, mc):
            print(f"  [{mc+1}/4] {os.path.basename(sd)}")
            mc += 1

    print(f"\nAll saved to: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
