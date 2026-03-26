"""
Phase 0: Linear Probe — 验证 VGGT patch tokens 是否包含目标检测信息

方法:
  1. 加载预训练 VGGT 模型 (仅 aggregator)
  2. 对 CA1M 场景提取 patch tokens [B, S, 1369, 2048]
  3. 将 GT 3D box 投影到 patch grid, 创建 token-level 二分类标签
  4. 训练一个 linear layer (2048 → 1) 做 objectness 分类
  5. 报告 AP 和 F1 等指标
"""

import os
import sys
import json
import glob
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, f1_score, accuracy_score

# Setup paths
VGGT_ROOT = "/home/fanwg/myproject/code/vggt"
sys.path.insert(0, VGGT_ROOT)
sys.path.insert(0, os.path.join(VGGT_ROOT, "training"))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# ─────────────────────────────────────────────
# 1. Data: 直接从 CA1M 目录读取, 不依赖训练数据管道
# ─────────────────────────────────────────────

CA1M_DIR = "/data1/lyq/CA1M-dataset/CA1M-dataset/training"
CA1M_ANNOTATION_DIR = "/data1/lyq/CA1M-dataset/CA1M-dataset"

IMG_SIZE = 518
PATCH_SIZE = 14
GRID_H = IMG_SIZE // PATCH_SIZE  # 37
GRID_W = IMG_SIZE // PATCH_SIZE  # 37


def load_scene_images(scene_dir, max_images=4):
    """加载场景的 RGB 图像, 返回预处理后的 tensor"""
    from vggt.utils.load_fn import load_and_preprocess_images_square

    rgb_dir = os.path.join(scene_dir, "rgb")
    img_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    if not img_files:
        img_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    if len(img_files) < 2:
        return None, None

    # 均匀采样
    step = max(1, len(img_files) // max_images)
    selected = img_files[::step][:max_images]

    images, _ = load_and_preprocess_images_square(selected, IMG_SIZE)
    return images, selected


def load_scene_annotations(scene_dir):
    """加载场景的相机参数和 3D bounding box"""
    # 内参
    K_file = os.path.join(scene_dir, "K_rgb.txt")
    if not os.path.exists(K_file):
        return None
    K = np.loadtxt(K_file).reshape(3, 3).astype(np.float32)

    # 外参 (all poses)
    poses_file = os.path.join(scene_dir, "all_poses.npy")
    if not os.path.exists(poses_file):
        return None
    all_poses = np.load(poses_file).astype(np.float32)  # [N, 4, 4]

    # 3D box corners
    corners_file = os.path.join(scene_dir, "after_filter_boxes.npy")
    if not os.path.exists(corners_file):
        return None
    bbox_corners = np.load(corners_file).astype(np.float32)  # [N_box, 8, 3] world coords

    return {"K": K, "all_poses": all_poses, "bbox_corners": bbox_corners}


def project_boxes_to_patch_grid(bbox_corners, K, pose, img_h, img_w):
    """
    将世界坐标系的 3D box 角点投影到 patch grid, 返回 token-level 标签

    Args:
        bbox_corners: [N_box, 8, 3] 世界坐标
        K: [3, 3] 内参
        pose: [4, 4] 外参 (world-to-camera)
        img_h, img_w: 图像尺寸
    Returns:
        labels: [GRID_H * GRID_W] 二分类标签 (0/1)
    """
    labels = np.zeros(GRID_H * GRID_W, dtype=np.float32)

    if bbox_corners is None or len(bbox_corners) == 0:
        return labels

    R = pose[:3, :3]  # [3, 3]
    t = pose[:3, 3]   # [3]

    for box in bbox_corners:
        # 世界坐标 → 相机坐标: p_cam = R @ p_world + t
        corners_cam = (R @ box.T).T + t  # [8, 3]

        # 过滤掉在相机后面的点
        valid = corners_cam[:, 2] > 0.1
        if valid.sum() < 4:
            continue

        corners_valid = corners_cam[valid]

        # 相机坐标 → 像素坐标
        px = K[0, 0] * corners_valid[:, 0] / corners_valid[:, 2] + K[0, 2]
        py = K[1, 1] * corners_valid[:, 1] / corners_valid[:, 2] + K[1, 2]

        # 计算 2D bounding rectangle
        x_min = max(0, int(np.floor(px.min())))
        x_max = min(img_w - 1, int(np.ceil(px.max())))
        y_min = max(0, int(np.floor(py.min())))
        y_max = min(img_h - 1, int(np.ceil(py.max())))

        if x_max <= x_min or y_max <= y_min:
            continue

        # 映射到 patch grid
        # 注意: load_and_preprocess_images_square 做了 square padding + resize
        # 这里简化处理, 假设图像已经是 IMG_SIZE x IMG_SIZE
        patch_x_min = max(0, x_min * GRID_W // img_w)
        patch_x_max = min(GRID_W - 1, x_max * GRID_W // img_w)
        patch_y_min = max(0, y_min * GRID_H // img_h)
        patch_y_max = min(GRID_H - 1, y_max * GRID_H // img_h)

        for py_idx in range(patch_y_min, patch_y_max + 1):
            for px_idx in range(patch_x_min, patch_x_max + 1):
                labels[py_idx * GRID_W + px_idx] = 1.0

    return labels


# ─────────────────────────────────────────────
# 2. Feature Extraction
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_features_and_labels(model, scene_dirs, max_scenes=200, imgs_per_scene=4):
    """提取 VGGT 特征和对应的 token-level 标签"""
    all_features = []
    all_labels = []
    success_count = 0

    for i, scene_dir in enumerate(scene_dirs):
        if success_count >= max_scenes:
            break

        # 加载图像
        images, img_paths = load_scene_images(scene_dir, max_images=imgs_per_scene)
        if images is None:
            continue

        # 加载标注
        anno = load_scene_annotations(scene_dir)
        if anno is None:
            continue

        K = anno["K"]
        all_poses = anno["all_poses"]
        bbox_corners = anno["bbox_corners"]

        # 获取选中图像对应的 pose indices
        all_rgb_files = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
        if not all_rgb_files:
            all_rgb_files = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.jpg")))

        selected_indices = []
        for p in img_paths:
            try:
                idx = all_rgb_files.index(p)
                selected_indices.append(idx)
            except ValueError:
                selected_indices.append(0)

        # VGGT 推理
        images_gpu = images.to("cuda")
        if images_gpu.dim() == 4:
            images_gpu = images_gpu.unsqueeze(0)  # [S,3,H,W] → [1,S,3,H,W]
        try:
            aggregated_tokens_list, patch_start_idx = model.aggregator(images_gpu)
        except Exception as e:
            log.warning(f"Scene {i} aggregator failed: {e}")
            continue

        # 提取最后一层的 patch tokens
        last_tokens = aggregated_tokens_list[-1]  # [B, S, P, 2048]
        B, S, P, C = last_tokens.shape
        patch_tokens = last_tokens[:, :, patch_start_idx:, :]  # [B, S, 1369, 2048]

        # 帧间平均 → consensus features
        consensus = patch_tokens.mean(dim=1).squeeze(0)  # [1369, 2048]
        consensus_np = consensus.cpu().float().numpy()

        # 创建 token-level 标签 (各帧标签取 union)
        combined_labels = np.zeros(GRID_H * GRID_W, dtype=np.float32)
        for frame_idx, sel_idx in enumerate(selected_indices):
            if sel_idx >= len(all_poses):
                continue
            pose = all_poses[sel_idx]  # [4, 4]
            # 使用原始图像尺寸进行投影 (用 K 对应的分辨率)
            # CA1M 的 K 对应原始分辨率, 这里用原始分辨率
            orig_h = int(2 * K[1, 2])
            orig_w = int(2 * K[0, 2])
            frame_labels = project_boxes_to_patch_grid(
                bbox_corners, K, pose, orig_h, orig_w
            )
            combined_labels = np.maximum(combined_labels, frame_labels)

        all_features.append(consensus_np)
        all_labels.append(combined_labels)
        success_count += 1

        if success_count % 20 == 0:
            pos_ratio = combined_labels.mean()
            log.info(f"[{success_count}/{max_scenes}] Scene {scene_dir.split('/')[-1]}, "
                     f"pos_ratio={pos_ratio:.3f}, tokens={consensus_np.shape}")

        # 释放 GPU 内存
        del images_gpu, aggregated_tokens_list, last_tokens, patch_tokens
        torch.cuda.empty_cache()

    # Stack
    features = np.concatenate(all_features, axis=0)  # [N_scenes * 1369, 2048]
    labels = np.concatenate(all_labels, axis=0)       # [N_scenes * 1369]

    log.info(f"Total samples: {len(features)}, positive ratio: {labels.mean():.4f}")
    return features, labels


# ─────────────────────────────────────────────
# 3. Linear Probe Training
# ─────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


def train_linear_probe(features, labels, epochs=20, lr=1e-3, batch_size=4096):
    """训练线性探测器"""
    # 划分 train/val (80/20)
    n = len(features)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    X_train = torch.from_numpy(features[train_idx]).cuda()
    y_train = torch.from_numpy(labels[train_idx]).cuda()
    X_val = torch.from_numpy(features[val_idx]).cuda()
    y_val = torch.from_numpy(labels[val_idx]).cuda()

    # 处理类别不平衡
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).cuda()
    log.info(f"Train: {len(X_train)} samples, pos_weight={pos_weight.item():.2f}")

    probe = LinearProbe(in_dim=features.shape[1]).cuda()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_ap = 0
    for epoch in range(epochs):
        probe.train()
        # Mini-batch training
        perm_train = torch.randperm(len(X_train))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))
            idx = perm_train[start:end]
            logits = probe(X_train[idx])
            loss = criterion(logits, y_train[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = []
            for start in range(0, len(X_val), batch_size):
                end = min(start + batch_size, len(X_val))
                val_logits.append(probe(X_val[start:end]))
            val_logits = torch.cat(val_logits)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_labels = y_val.cpu().numpy()

            ap = average_precision_score(val_labels, val_probs)
            preds_binary = (val_probs > 0.5).astype(float)
            f1 = f1_score(val_labels, preds_binary, zero_division=0)
            acc = accuracy_score(val_labels, preds_binary)

        if ap > best_ap:
            best_ap = ap

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info(f"Epoch {epoch+1:3d} | loss={total_loss/n_batches:.4f} | "
                     f"AP={ap:.4f} | F1={f1:.4f} | Acc={acc:.4f} | best_AP={best_ap:.4f}")

    log.info(f"\n{'='*60}")
    log.info(f"FINAL RESULT: Best AP = {best_ap:.4f}")
    log.info(f"{'='*60}")
    log.info(f"AP > 0.5 → VGGT 特征包含目标检测信息, 方案可行")
    log.info(f"AP > 0.3 → 有一定信息, 可能需要多层特征或 unfreeze aggregator")
    log.info(f"AP < 0.3 → 信息不足, 需要重新考虑方案")

    return best_ap


# ─────────────────────────────────────────────
# 4. 额外验证: 多层特征对比
# ─────────────────────────────────────────────

def extract_multilayer_features(model, scene_dirs, max_scenes=100, imgs_per_scene=4,
                                layer_indices=None):
    """提取多个中间层的特征, 对比不同层的检测信息量"""
    if layer_indices is None:
        layer_indices = [3, 11, 17, 23]  # 对应 DPTHead 使用的层

    all_features_by_layer = {l: [] for l in layer_indices}
    all_labels = []
    success_count = 0

    for i, scene_dir in enumerate(scene_dirs):
        if success_count >= max_scenes:
            break

        images, img_paths = load_scene_images(scene_dir, max_images=imgs_per_scene)
        if images is None:
            continue

        anno = load_scene_annotations(scene_dir)
        if anno is None:
            continue

        K = anno["K"]
        all_poses = anno["all_poses"]
        bbox_corners = anno["bbox_corners"]

        all_rgb_files = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
        if not all_rgb_files:
            all_rgb_files = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.jpg")))

        selected_indices = []
        for p in img_paths:
            try:
                idx = all_rgb_files.index(p)
                selected_indices.append(idx)
            except ValueError:
                selected_indices.append(0)

        images_gpu = images.to("cuda")
        if images_gpu.dim() == 4:
            images_gpu = images_gpu.unsqueeze(0)
        try:
            with torch.no_grad():
                aggregated_tokens_list, patch_start_idx = model.aggregator(images_gpu)
        except Exception as e:
            continue

        # 提取各层特征
        for layer_idx in layer_indices:
            tokens = aggregated_tokens_list[layer_idx]  # [B, S, P, 2048]
            patch_tokens = tokens[:, :, patch_start_idx:, :]
            consensus = patch_tokens.mean(dim=1).squeeze(0).cpu().float().numpy()
            all_features_by_layer[layer_idx].append(consensus)

        # 创建标签 (same as before)
        combined_labels = np.zeros(GRID_H * GRID_W, dtype=np.float32)
        for frame_idx, sel_idx in enumerate(selected_indices):
            if sel_idx >= len(all_poses):
                continue
            pose = all_poses[sel_idx]
            orig_h = int(2 * K[1, 2])
            orig_w = int(2 * K[0, 2])
            frame_labels = project_boxes_to_patch_grid(
                bbox_corners, K, pose, orig_h, orig_w
            )
            combined_labels = np.maximum(combined_labels, frame_labels)

        all_labels.append(combined_labels)
        success_count += 1

        del images_gpu, aggregated_tokens_list
        torch.cuda.empty_cache()

    labels = np.concatenate(all_labels, axis=0)

    log.info(f"\n{'='*60}")
    log.info(f"多层特征 Linear Probe 对比 (共 {success_count} 场景)")
    log.info(f"{'='*60}")

    for layer_idx in layer_indices:
        features = np.concatenate(all_features_by_layer[layer_idx], axis=0)
        log.info(f"\n--- Layer {layer_idx} (features shape: {features.shape}) ---")
        ap = train_linear_probe(features, labels, epochs=15, lr=1e-3)

    return


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Phase 0: VGGT Feature Linear Probe for Object Detection")
    log.info("=" * 60)

    # 1. 加载模型
    log.info("Loading VGGT model (aggregator only)...")
    model = VGGT(
        enable_camera=False, enable_gravity=False, enable_depth=False,
        enable_point=False, enable_track=False, enable_cubify=False
    )
    ckpt = torch.load("/home/lanyuqing/model/model.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    log.info(f"Model loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
    model = model.cuda().eval()

    # 2. 获取场景列表
    scene_dirs = sorted(glob.glob(os.path.join(CA1M_DIR, "*")))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
    log.info(f"Found {len(scene_dirs)} scenes in CA1M training set")

    # 随机打乱
    np.random.seed(42)
    np.random.shuffle(scene_dirs)

    # 3. 实验 1 已完成: AP=0.9664, 跳过
    # log.info("\n--- 实验 1: 最后一层特征 (Layer 23) ---")
    # features, labels = extract_features_and_labels(
    #     model, scene_dirs, max_scenes=200, imgs_per_scene=4
    # )
    # best_ap = train_linear_probe(features, labels, epochs=20, lr=1e-3)

    # 4. 多层对比
    log.info("\n--- 实验 2: 多层特征对比 (Layers 3, 11, 17, 23) ---")
    extract_multilayer_features(
        model, scene_dirs, max_scenes=100, imgs_per_scene=4,
        layer_indices=[3, 11, 17, 23]
    )

    log.info("\nPhase 0 完成!")


if __name__ == "__main__":
    from vggt.models.vggt import VGGT
    main()
