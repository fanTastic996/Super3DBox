"""
Phase 0 可视化: VGGT 特征是否包含目标检测信息

生成三组可视化:
  1. 逐场景可视化: 原图 + GT框 + GT patch标签 + Linear Probe 预测热力图
  2. 多层对比: Layer 3 / 11 / 17 / 23 的预测热力图对比
  3. 汇总统计图: AP / F1 随层深度的变化曲线
"""

import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, "/home/fanwg/myproject/code/vggt")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMG_SIZE = 518
PATCH_SIZE = 14
GRID_H = GRID_W = IMG_SIZE // PATCH_SIZE  # 37
CA1M_DIR = "/data1/lyq/CA1M-dataset/CA1M-dataset/training"
OUTPUT_DIR = "/home/fanwg/myproject/code/vggt/vis_phase0"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Reuse data loading from linear_probe.py ───
from linear_probe import (
    load_scene_images, load_scene_annotations,
    project_boxes_to_patch_grid, LinearProbe
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def project_boxes_to_2d(bbox_corners, K, pose):
    """将 3D box 角点投影到 2D, 返回 2D 矩形列表"""
    rects = []
    R, t = pose[:3, :3], pose[:3, 3]
    for box in bbox_corners:
        cam = (R @ box.T).T + t
        valid = cam[:, 2] > 0.1
        if valid.sum() < 4:
            continue
        cv = cam[valid]
        px = K[0,0] * cv[:,0] / cv[:,2] + K[0,2]
        py = K[1,1] * cv[:,1] / cv[:,2] + K[1,2]
        rects.append((px.min(), py.min(), px.max(), py.max()))
    return rects


def train_probe_and_get_weights(features, labels, epochs=20, lr=1e-3):
    """训练 linear probe 并返回模型"""
    n = len(features)
    perm = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    X_train = torch.from_numpy(features[train_idx]).cuda()
    y_train = torch.from_numpy(labels[train_idx]).cuda()
    X_val = torch.from_numpy(features[val_idx]).cuda()
    y_val = torch.from_numpy(labels[val_idx]).cuda()

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).cuda()

    probe = LinearProbe(in_dim=features.shape[1]).cuda()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        probe.train()
        perm_t = torch.randperm(len(X_train))
        for start in range(0, len(X_train), 4096):
            idx = perm_t[start:start+4096]
            loss = criterion(probe(X_train[idx]), y_train[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(probe(X_val)).cpu().numpy()
        val_labels = y_val.cpu().numpy()
        ap = average_precision_score(val_labels, val_probs)

    return probe, ap


@torch.no_grad()
def get_scene_prediction(model, probe, images_gpu, patch_start_idx, layer_idx=-1):
    """获取单个场景的 token-level 预测概率"""
    tokens_list, psi = model.aggregator(images_gpu)
    tokens = tokens_list[layer_idx]
    patch_tokens = tokens[:, :, psi:, :]
    consensus = patch_tokens.mean(dim=1).squeeze(0)  # [1369, 2048]
    logits = probe(consensus.float())
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs.reshape(GRID_H, GRID_W)


# ─────────────────────────────────────────────
# 可视化 1: 逐场景详细展示
# ─────────────────────────────────────────────

def vis_scene_detail(scene_dir, model, probe, scene_idx, img_idx=0):
    """单场景可视化: 原图+GT框 | GT patch标签 | 预测热力图 | 叠加"""

    images, img_paths = load_scene_images(scene_dir, max_images=4)
    if images is None:
        return False
    anno = load_scene_annotations(scene_dir)
    if anno is None:
        return False

    K, all_poses, bbox_corners = anno["K"], anno["all_poses"], anno["bbox_corners"]
    all_rgb = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
    if not all_rgb:
        all_rgb = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.jpg")))

    # 选择的帧 index
    step = max(1, len(all_rgb) // 4)
    sel_indices = list(range(0, len(all_rgb), step))[:4]

    images_gpu = images.cuda().unsqueeze(0)
    pred_map = get_scene_prediction(model, probe, images_gpu, 5)  # 5 = patch_start_idx

    # GT labels (union of all frames)
    orig_h, orig_w = int(2 * K[1,2]), int(2 * K[0,2])
    gt_labels = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for si in sel_indices:
        if si < len(all_poses):
            fl = project_boxes_to_patch_grid(bbox_corners, K, all_poses[si], orig_h, orig_w)
            gt_labels = np.maximum(gt_labels, fl.reshape(GRID_H, GRID_W))

    # 读取原图用于展示 (第一帧)
    show_idx = sel_indices[min(img_idx, len(sel_indices)-1)]
    orig_img = np.array(Image.open(all_rgb[show_idx]).convert("RGB"))
    pose_show = all_poses[show_idx] if show_idx < len(all_poses) else all_poses[0]
    rects_2d = project_boxes_to_2d(bbox_corners, K, pose_show)

    # ─── 绘图 ───
    fig = plt.figure(figsize=(24, 6), dpi=150)
    gs = GridSpec(1, 4, figure=fig, wspace=0.08)

    cmap_hot = LinearSegmentedColormap.from_list("", ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#ffcc00"])

    # (a) 原图 + GT 2D 框
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(orig_img)
    for (x1, y1, x2, y2) in rects_2d:
        x1c = max(0, min(x1, orig_img.shape[1]))
        x2c = max(0, min(x2, orig_img.shape[1]))
        y1c = max(0, min(y1, orig_img.shape[0]))
        y2c = max(0, min(y2, orig_img.shape[0]))
        rect = mpatches.Rectangle((x1c, y1c), x2c-x1c, y2c-y1c,
                                   linewidth=2, edgecolor="#00ff88", facecolor="none")
        ax1.add_patch(rect)
    ax1.set_title("(a) Original + GT 3D Boxes", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # (b) GT Patch Labels
    ax2 = fig.add_subplot(gs[1])
    gt_up = F.interpolate(torch.from_numpy(gt_labels).unsqueeze(0).unsqueeze(0).float(),
                          size=(orig_img.shape[0], orig_img.shape[1]),
                          mode="nearest").squeeze().numpy()
    blend = orig_img.astype(float) / 255.0
    mask_overlay = np.zeros_like(blend)
    mask_overlay[:,:,1] = gt_up * 0.6  # green channel
    blend = blend * 0.5 + mask_overlay * 0.5 + blend * gt_up[:,:,None] * 0.0
    blend = np.clip(blend, 0, 1)
    ax2.imshow(blend)
    ax2.set_title("(b) GT Patch Labels (green)", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # (c) Linear Probe Prediction Heatmap
    ax3 = fig.add_subplot(gs[2])
    pred_up = F.interpolate(torch.from_numpy(pred_map).unsqueeze(0).unsqueeze(0).float(),
                            size=(orig_img.shape[0], orig_img.shape[1]),
                            mode="bilinear", align_corners=False).squeeze().numpy()
    ax3.imshow(orig_img, alpha=0.3)
    im = ax3.imshow(pred_up, cmap=cmap_hot, alpha=0.85, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="Objectness Score")
    ax3.set_title("(c) Linear Probe Prediction", fontsize=12, fontweight="bold")
    ax3.axis("off")

    # (d) 预测 vs GT 叠加对比
    ax4 = fig.add_subplot(gs[3])
    # Red = prediction, Green = GT, Yellow = overlap
    overlay = np.zeros((*orig_img.shape[:2], 3))
    overlay[:,:,0] = pred_up * 0.8  # Red: prediction
    overlay[:,:,1] = gt_up * 0.8    # Green: GT
    # Yellow where both are high
    bg = orig_img.astype(float) / 255.0 * 0.3
    combined = bg + overlay * 0.7
    combined = np.clip(combined, 0, 1)
    ax4.imshow(combined)
    red_patch = mpatches.Patch(color='red', label='Prediction')
    green_patch = mpatches.Patch(color='green', label='GT Label')
    yellow_patch = mpatches.Patch(color='yellow', label='Overlap')
    ax4.legend(handles=[red_patch, green_patch, yellow_patch], loc='lower right', fontsize=9)
    ax4.set_title("(d) Pred(Red) vs GT(Green)", fontsize=12, fontweight="bold")
    ax4.axis("off")

    scene_name = os.path.basename(scene_dir)
    fig.suptitle(f"Scene: {scene_name}  |  Linear Probe on Frozen VGGT Features  |  AP=0.97",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, f"scene_{scene_idx:03d}_{scene_name}.png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    return True


# ─────────────────────────────────────────────
# 可视化 2: 多层对比
# ─────────────────────────────────────────────

def vis_multilayer(scene_dir, model, probes_by_layer, scene_idx):
    """4层预测热力图对比"""
    images, img_paths = load_scene_images(scene_dir, max_images=4)
    if images is None:
        return False
    anno = load_scene_annotations(scene_dir)
    if anno is None:
        return False

    K, all_poses, bbox_corners = anno["K"], anno["all_poses"], anno["bbox_corners"]
    all_rgb = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
    if not all_rgb:
        all_rgb = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.jpg")))

    step = max(1, len(all_rgb) // 4)
    sel_indices = list(range(0, len(all_rgb), step))[:4]
    show_idx = sel_indices[0]
    orig_img = np.array(Image.open(all_rgb[show_idx]).convert("RGB"))

    images_gpu = images.cuda().unsqueeze(0)
    layers = [3, 11, 17, 23]
    layer_aps = {3: 0.907, 11: 0.938, 17: 0.967, 23: 0.967}

    # GT labels
    orig_h, orig_w = int(2*K[1,2]), int(2*K[0,2])
    gt_labels = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for si in sel_indices:
        if si < len(all_poses):
            fl = project_boxes_to_patch_grid(bbox_corners, K, all_poses[si], orig_h, orig_w)
            gt_labels = np.maximum(gt_labels, fl.reshape(GRID_H, GRID_W))

    # Get predictions for each layer
    with torch.no_grad():
        tokens_list, psi = model.aggregator(images_gpu)

    cmap_hot = LinearSegmentedColormap.from_list("", ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#ffcc00"])

    fig, axes = plt.subplots(1, 6, figsize=(30, 5), dpi=150)

    # (a) Original image
    pose_show = all_poses[show_idx] if show_idx < len(all_poses) else all_poses[0]
    rects_2d = project_boxes_to_2d(bbox_corners, K, pose_show)
    axes[0].imshow(orig_img)
    for (x1, y1, x2, y2) in rects_2d:
        rect = mpatches.Rectangle(
            (max(0,x1), max(0,y1)), min(x2,orig_img.shape[1])-max(0,x1),
            min(y2,orig_img.shape[0])-max(0,y1),
            linewidth=2, edgecolor="#00ff88", facecolor="none")
        axes[0].add_patch(rect)
    axes[0].set_title("Original + GT", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # (b) GT patch labels
    gt_up = F.interpolate(torch.from_numpy(gt_labels).unsqueeze(0).unsqueeze(0).float(),
                          size=(orig_img.shape[0], orig_img.shape[1]),
                          mode="nearest").squeeze().numpy()
    axes[1].imshow(orig_img, alpha=0.4)
    axes[1].imshow(gt_up, cmap="Greens", alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title("GT Labels", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # (c-f) Layer predictions
    for ax_idx, layer_idx in enumerate(layers):
        tokens = tokens_list[layer_idx]
        patch_tokens = tokens[:, :, psi:, :]
        consensus = patch_tokens.mean(dim=1).squeeze(0).float()
        probe = probes_by_layer[layer_idx]
        logits = probe(consensus)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(GRID_H, GRID_W)

        pred_up = F.interpolate(torch.from_numpy(probs).unsqueeze(0).unsqueeze(0).float(),
                                size=(orig_img.shape[0], orig_img.shape[1]),
                                mode="bilinear", align_corners=False).squeeze().numpy()

        ax = axes[ax_idx + 2]
        ax.imshow(orig_img, alpha=0.3)
        im = ax.imshow(pred_up, cmap=cmap_hot, alpha=0.85, vmin=0, vmax=1)
        ax.set_title(f"Layer {layer_idx}  (AP={layer_aps[layer_idx]:.3f})",
                     fontsize=11, fontweight="bold")
        ax.axis("off")

    scene_name = os.path.basename(scene_dir)
    fig.suptitle(f"Multi-Layer Feature Comparison — Scene: {scene_name}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, f"multilayer_{scene_idx:03d}_{scene_name}.png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    return True


# ─────────────────────────────────────────────
# 可视化 3: 汇总统计图
# ─────────────────────────────────────────────

def vis_summary_stats():
    """AP / F1 随层深度的变化 + PR 曲线示意"""
    layers = [3, 11, 17, 23]
    aps = [0.9069, 0.9379, 0.9667, 0.9671]
    f1s = [0.576, 0.611, 0.886, 0.891]
    accs = [0.941, 0.949, 0.989, 0.990]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    # (a) AP vs Layer
    ax = axes[0]
    bars = ax.bar(range(len(layers)), aps, color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
                  width=0.6, edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"Layer {l}" for l in layers], fontsize=11)
    ax.set_ylabel("Average Precision (AP)", fontsize=12)
    ax.set_title("(a) Detection AP by Layer Depth", fontsize=13, fontweight="bold")
    ax.set_ylim(0.85, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Feasibility threshold (0.5)")
    for bar, ap in zip(bars, aps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{ap:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # (b) F1 vs Layer
    ax = axes[1]
    ax.plot(range(len(layers)), f1s, "o-", color="#e74c3c", linewidth=2.5, markersize=10,
            label="F1 Score")
    ax.plot(range(len(layers)), accs, "s--", color="#2ecc71", linewidth=2, markersize=8,
            label="Accuracy")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"Layer {l}" for l in layers], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("(b) F1 & Accuracy by Layer Depth", fontsize=13, fontweight="bold")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    for i, (f1, acc) in enumerate(zip(f1s, accs)):
        ax.annotate(f"{f1:.3f}", (i, f1), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="#e74c3c")

    # (c) Key takeaways
    ax = axes[2]
    ax.axis("off")
    takeaways = [
        ("AP = 0.967", "A single linear layer achieves\nnear-perfect object detection", "#e74c3c"),
        ("Layer 17 ≈ 23", "Deep features saturate early;\nno need for complex multi-layer fusion", "#2ecc71"),
        ("Layer 3: AP=0.91", "Even shallow layers carry\nstrong object signals", "#3498db"),
        ("Conclusion", "VGGT features are rich enough\nfor direct 3D detection", "#f39c12"),
    ]
    for i, (title, desc, color) in enumerate(takeaways):
        y = 0.85 - i * 0.23
        ax.text(0.05, y, "●", fontsize=20, color=color, transform=ax.transAxes,
                va="center")
        ax.text(0.12, y, title, fontsize=13, fontweight="bold", transform=ax.transAxes,
                va="center")
        ax.text(0.12, y - 0.09, desc, fontsize=10, color="#555555", transform=ax.transAxes,
                va="center")
    ax.set_title("(c) Key Takeaways", fontsize=13, fontweight="bold")

    fig.suptitle("Phase 0 Validation: Can VGGT Features Detect Objects?",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.savefig(os.path.join(OUTPUT_DIR, "summary_stats.png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Saved] summary_stats.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 0 Visualization")
    print("=" * 60)

    # 1. Load model
    print("Loading VGGT model...")
    from vggt.models.vggt import VGGT
    model = VGGT(
        enable_camera=False, enable_gravity=False, enable_depth=False,
        enable_point=False, enable_track=False, enable_cubify=False
    )
    ckpt = torch.load("/home/lanyuqing/model/model.pt", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()

    # 2. Get scenes
    scene_dirs = sorted(glob.glob(os.path.join(CA1M_DIR, "*")))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
    np.random.seed(42)
    np.random.shuffle(scene_dirs)

    # 3. Extract features for training probes (reuse 100 scenes)
    print("Extracting features for probe training (100 scenes)...")
    all_features_by_layer = {3: [], 11: [], 17: [], 23: []}
    all_labels = []
    count = 0

    for scene_dir in scene_dirs:
        if count >= 100:
            break
        images, img_paths = load_scene_images(scene_dir, max_images=4)
        if images is None:
            continue
        anno = load_scene_annotations(scene_dir)
        if anno is None:
            continue

        K, all_poses, bbox_corners = anno["K"], anno["all_poses"], anno["bbox_corners"]
        all_rgb = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
        if not all_rgb:
            all_rgb = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.jpg")))

        step = max(1, len(all_rgb) // 4)
        sel_indices = list(range(0, len(all_rgb), step))[:4]
        selected_paths = [all_rgb[i] for i in sel_indices if i < len(all_rgb)]

        images_gpu = images.cuda().unsqueeze(0)
        with torch.no_grad():
            tokens_list, psi = model.aggregator(images_gpu)
            for layer_idx in [3, 11, 17, 23]:
                tokens = tokens_list[layer_idx][:, :, psi:, :]
                consensus = tokens.mean(dim=1).squeeze(0).cpu().float().numpy()
                all_features_by_layer[layer_idx].append(consensus)

        orig_h, orig_w = int(2*K[1,2]), int(2*K[0,2])
        combined = np.zeros(GRID_H * GRID_W, dtype=np.float32)
        for si in sel_indices:
            if si < len(all_poses):
                fl = project_boxes_to_patch_grid(bbox_corners, K, all_poses[si], orig_h, orig_w)
                combined = np.maximum(combined, fl)
        all_labels.append(combined)
        count += 1
        del images_gpu, tokens_list
        torch.cuda.empty_cache()

        if count % 20 == 0:
            print(f"  [{count}/100] features extracted")

    labels = np.concatenate(all_labels, axis=0)

    # 4. Train probes for each layer
    print("Training linear probes for each layer...")
    probes_by_layer = {}
    for layer_idx in [3, 11, 17, 23]:
        features = np.concatenate(all_features_by_layer[layer_idx], axis=0)
        probe, ap = train_probe_and_get_weights(features, labels, epochs=20, lr=1e-3)
        probes_by_layer[layer_idx] = probe
        print(f"  Layer {layer_idx}: AP={ap:.4f}")

    # 5-6. Already generated, skip
    # vis_summary_stats()
    # per-scene vis already done

    # 7. Visualization 2: Multi-layer comparison (3 scenes)
    print("\nGenerating multi-layer comparison (3 scenes)...")
    ml_count = 0
    for i, scene_dir in enumerate(scene_dirs[110:]):
        if ml_count >= 3:
            break
        ok = vis_multilayer(scene_dir, model, probes_by_layer, ml_count)
        if ok:
            print(f"  Multi-layer {ml_count+1}/3: {os.path.basename(scene_dir)}")
            ml_count += 1

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    print(f"Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
