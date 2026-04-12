"""
Single-scene overfit training for QCOD Stage A.

Fixed scene: 42898849 (test split)
Fixed frames: [186, 361, 569, 828]

Trains ONLY the QCOD modules (query_evolution + qcod_head) while all other
components (aggregator, camera/depth/gravity heads) are frozen.

Every N steps, exports predictions to .pkl for offline visualization.

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n superbox python scripts/qcod_overfit_single_scene.py
"""

import os
import sys
import types
import pickle
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))
sys.path.insert(0, REPO_ROOT)

# Stub open3d
if "open3d" not in sys.modules:
    o3d_stub = types.ModuleType("open3d")
    o3d_stub.io = types.SimpleNamespace(read_point_cloud=lambda *a, **k: None)
    sys.modules["open3d"] = o3d_stub

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from omegaconf import OmegaConf

# ── Config ──────────────────────────────────────────────────────────────
# SCENE_DIR = "/data1/fanwg/CA1M-dataset/CA1M-dataset/test/42898849"
# SEQ_NAME = "42898849"
# IMAGE_IDXS = [186, 361, 569, 828]

# SCENE_DIR = "/data1/fanwg/CA1M-dataset/CA1M-dataset/test/47115525"
# SEQ_NAME = "47115525"
# IMAGE_IDXS = [144, 156, 164, 238]

# SCENE_DIR = "/data1/fanwg/CA1M-dataset/CA1M-dataset/test/47204573"
# SEQ_NAME = "47204573"
# IMAGE_IDXS = [101, 409, 444, 505]

SCENE_DIR = "/data1/fanwg/CA1M-dataset/CA1M-dataset/test/47333452"
SEQ_NAME = "47333452"
IMAGE_IDXS = [70, 137, 167, 214]

NUM_STEPS = 1000
LR = 5e-4
LOG_EVERY = 100
EXPORT_EVERY = 200
OUTPUT_DIR = os.path.join(REPO_ROOT, "pkl_output", f"overfit_vol_weighted_{SEQ_NAME}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_common_config():
    return OmegaConf.create({
        "fix_img_num": -1, "fix_aspect_ratio": 1.0,
        "load_track": False, "track_num": 16, "training": True,
        "inside_random": False, "img_size": 518, "patch_size": 14,
        "rescale": True, "rescale_aug": False, "landscape_check": False,
        "debug": False, "get_nearby": True, "load_depth": True,
        "img_nums": [4, 4], "allow_duplicate_img": True, "repeat_batch": False,
        "augs": {"use": False, "cojitter": False, "cojitter_ratio": 0.5,
                 "scales": None, "aspects": [1.0, 1.0], "color_jitter": None,
                 "gray_scale": False, "gau_blur": False},
    })


def load_fixed_batch():
    """Load a single fixed batch from the specified scene + frames.

    Returns:
        batch: dict of tensors (for training)
        vis_metadata: dict with UUID-based visibility info (for pkl export)
    """
    from data.datasets.ca1m import CA1MDataset

    ds = CA1MDataset(
        common_conf=make_common_config(),
        split="test",
        CA1M_DIR="/data1/fanwg/CA1M-dataset/CA1M-dataset/test/",
        CA1M_ANNOTATION_DIR="/data1/fanwg/CA1M-dataset/CA1M-dataset/",
    )

    print("  Loading scene data + building fixed-frame batch...")
    return _load_batch_manual(ds, SEQ_NAME, IMAGE_IDXS)


def _load_batch_manual(ds, seq_name, image_idxs):
    """Manually load a batch for specific frames, bypassing random sampling."""
    import cv2
    from data.dataset_util import (
        read_image_cv2, read_depth, threshold_depth_map,
        load_gt_corners_cam_multiframe, load_scene_gt_in_first_frame,
    )
    from vggt.utils.geometry import closed_form_inverse_se3
    from vggt.utils.rotation import mat_to_quat

    scene_path = os.path.join(ds.CA1M_DIR, seq_name)
    scene_data = ds.load_scene_data(seq_name)

    seq_poses = scene_data['poses'].reshape(-1, 4, 4)
    seq_poses = closed_form_inverse_se3(
        torch.from_numpy(seq_poses.astype(np.float32))
    ).numpy()

    K_rgb = scene_data['K']
    seq_gravity = scene_data['gravity']
    json_directory = os.path.join(ds.CA1M_DIR, seq_name, 'instances')

    target_shape = ds.get_target_shape(1.0)

    images, depths, extrinsics, intrinsics = [], [], [], []
    cam_points, world_points, point_masks = [], [], []
    all_gravity, original_sizes = [], []

    for img_idx in image_idxs:
        image_path = os.path.join(ds.CA1M_DIR, seq_name, 'rgb', f'{img_idx}.png')
        image = read_image_cv2(image_path)
        depth_path = image_path.replace("/rgb", "/depth")
        depth_map = read_depth(depth_path, 0.001)
        mvs_mask = np.ones_like(depth_map, dtype=bool)
        depth_map[~mvs_mask] = 0
        depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)

        original_size = np.array(image.shape[:2])
        if original_size[0] != depth_map.shape[0] or original_size[1] != depth_map.shape[1]:
            image = cv2.resize(image, (depth_map.shape[1], depth_map.shape[0]),
                               interpolation=cv2.INTER_AREA)
            original_size = np.array(image.shape[:2])

        cur_pose = seq_poses[img_idx]
        extri_opencv = cur_pose[:3, :]
        intri_opencv = K_rgb.copy()

        filepath = os.path.join(seq_name, 'rgb', f'{img_idx}.png')
        (image, depth_map, extri_opencv, intri_opencv,
         world_coords_points, cam_coords_points, point_mask, _) = \
            ds.process_one_image(
                image, depth_map, extri_opencv, intri_opencv,
                original_size, target_shape, filepath=filepath)

        images.append(image)
        depths.append(depth_map)
        extrinsics.append(extri_opencv)
        intrinsics.append(intri_opencv)
        cam_points.append(cam_coords_points)
        world_points.append(world_coords_points)
        point_masks.append(point_mask)
        original_sizes.append(original_size)

        gravity = seq_gravity[img_idx]
        all_gravity.append(gravity)

    # Per-frame box corners
    corners_cam_list = load_gt_corners_cam_multiframe(
        json_directory, image_idxs, json_name_fmt="{idx}.json", device="cpu"
    )
    for i in range(len(corners_cam_list)):
        if isinstance(corners_cam_list[i], tuple):
            corners_cam_list[i] = np.zeros((1, 8, 3), dtype=np.float32)
        corners_cam_list[i] = ds.process_bbox_corners(corners_cam_list[i])
    corners_cam_list = np.stack(corners_cam_list, axis=0).reshape(-1, 8, 3)

    # QCOD scene GT
    first_frame_idx = int(image_idxs[0])
    first_frame_w2c = seq_poses[first_frame_idx].astype(np.float32)
    scene_gt = load_scene_gt_in_first_frame(
        scene_path=scene_path,
        first_frame_w2c=first_frame_w2c,
        first_frame_idx_for_check=first_frame_idx,
        sanity_check=True,
    )

    # ── Visibility filter (plan: UUID-based cross-frame visibility) ──
    # Read each per-frame instances/{frame}.json, collect the UUID set,
    # count how many of the 4 sampled frames each scene instance appears in.
    # Keep only scene instances visible in >= VIS_MIN frames.
    #
    # Rationale: with O=64 queries and scene N=403, Hungarian matching was
    # oscillating because each step only 64/403 instances could match. By
    # filtering to instances actually observable in our 4 frames (and ideally
    # seen from multiple views for cross-view consistency), we reduce N to
    # ~63 — a near-perfect match for O=64, stabilizing training.
    VIS_MIN = 2
    import json as _json

    per_frame_visible_ids = {}
    per_frame_boxes_by_uuid = {}  # {frame_idx: {uuid: np.ndarray [8,3]}}
    for img_idx in image_idxs:
        pf_path = os.path.join(scene_path, "instances", f"{img_idx}.json")
        ids_this_frame = set()
        boxes_this_frame = {}
        if os.path.exists(pf_path):
            with open(pf_path, "r") as f:
                pf_data = _json.load(f)
            for inst in pf_data:
                if inst.get("category") in ["wall", "floor", "ceiling"]:
                    continue
                iid = inst.get("id")
                if iid is None:
                    continue
                ids_this_frame.add(iid)
                try:
                    boxes_this_frame[iid] = np.asarray(
                        inst["corners"], dtype=np.float32
                    ).reshape(8, 3)
                except (KeyError, ValueError):
                    pass
        per_frame_visible_ids[int(img_idx)] = ids_this_frame
        per_frame_boxes_by_uuid[int(img_idx)] = boxes_this_frame

    # Count visibility per scene-instance
    scene_ids_list = scene_gt["ids"]  # List[str]
    vis_count_per_scene = np.zeros(len(scene_ids_list), dtype=np.int32)
    vis_frames_per_scene = [[] for _ in scene_ids_list]
    for i, sid in enumerate(scene_ids_list):
        for f in image_idxs:
            if sid in per_frame_visible_ids[int(f)]:
                vis_count_per_scene[i] += 1
                vis_frames_per_scene[i].append(int(f))

    # Build keep-mask
    keep_mask = vis_count_per_scene >= VIS_MIN
    n_kept = int(keep_mask.sum())
    n_total = len(scene_ids_list)
    print(
        f"  [QCOD vis-filter] keeping {n_kept}/{n_total} scene instances "
        f"(vis_count >= {VIS_MIN})"
    )

    filtered_corners = scene_gt["corners"][keep_mask]
    filtered_R = scene_gt["R"][keep_mask]
    filtered_scale = scene_gt["scale"][keep_mask]
    filtered_ids = [scene_ids_list[i] for i in range(n_total) if keep_mask[i]]
    filtered_vis_count = vis_count_per_scene[keep_mask]
    filtered_vis_frames = [vis_frames_per_scene[i] for i in range(n_total) if keep_mask[i]]

    qcod_n_max = 500
    n_real = min(n_kept, qcod_n_max)
    qcod_scene_corners = np.zeros((qcod_n_max, 8, 3), dtype=np.float32)
    qcod_scene_R = np.broadcast_to(np.eye(3, dtype=np.float32), (qcod_n_max, 3, 3)).copy()
    qcod_scene_scale = np.zeros((qcod_n_max, 3), dtype=np.float32)
    qcod_scene_valid = np.zeros((qcod_n_max,), dtype=bool)
    if n_real > 0:
        qcod_scene_corners[:n_real] = filtered_corners[:n_real]
        qcod_scene_R[:n_real] = filtered_R[:n_real]
        qcod_scene_scale[:n_real] = filtered_scale[:n_real]
        qcod_scene_valid[:n_real] = True

    # Build batch dict (numpy)
    raw_batch = {
        "seq_name": f"CA1M_{seq_name}",
        "ids": np.array(image_idxs),
        "frame_num": len(extrinsics),
        "images": images,
        "depths": depths,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "cam_points": cam_points,
        "world_points": world_points,
        "bbox_corners": corners_cam_list,
        "point_masks": point_masks,
        "original_sizes": original_sizes,
        "gravity": all_gravity,
        "qcod_scene_corners": qcod_scene_corners,
        "qcod_scene_R": qcod_scene_R,
        "qcod_scene_scale": qcod_scene_scale,
        "qcod_scene_valid": qcod_scene_valid,
    }

    # Convert to tensors (same as ComposedDataset.__getitem__)
    sample = {
        "seq_name": raw_batch["seq_name"],
        "ids": torch.from_numpy(raw_batch["ids"]),
        "images": torch.from_numpy(np.stack(raw_batch["images"]).astype(np.float32)).permute(0, 3, 1, 2).div(255),
        "depths": torch.from_numpy(np.stack(raw_batch["depths"]).astype(np.float32)),
        "extrinsics": torch.from_numpy(np.stack(raw_batch["extrinsics"]).astype(np.float32)),
        "intrinsics": torch.from_numpy(np.stack(raw_batch["intrinsics"]).astype(np.float32)),
        "cam_points": torch.from_numpy(np.stack(raw_batch["cam_points"]).astype(np.float32)),
        "world_points": torch.from_numpy(np.stack(raw_batch["world_points"]).astype(np.float32)),
        "bbox_corners": torch.from_numpy(raw_batch["bbox_corners"].astype(np.float32)),
        "point_masks": torch.from_numpy(np.stack(raw_batch["point_masks"])),
        "gravity": mat_to_quat(torch.from_numpy(np.stack(raw_batch["gravity"]).astype(np.float32))),
        "qcod_scene_corners": torch.from_numpy(raw_batch["qcod_scene_corners"]),
        "qcod_scene_R": torch.from_numpy(raw_batch["qcod_scene_R"]),
        "qcod_scene_scale": torch.from_numpy(raw_batch["qcod_scene_scale"]),
        "qcod_scene_valid": torch.from_numpy(raw_batch["qcod_scene_valid"]),
    }

    # Collate as batch of 1
    batch = default_collate([{k: v for k, v in sample.items() if v is not None}])

    # Normalize via Trainer._process_batch
    from trainer import Trainer
    fake_self = types.SimpleNamespace(
        data_conf=types.SimpleNamespace(
            train=types.SimpleNamespace(
                common_config=types.SimpleNamespace(repeat_batch=False)
            )
        )
    )
    batch = Trainer._process_batch(fake_self, batch)

    # Return metadata for the visibility filter & pkl export
    # (can't be tensors — they're Python structures with string UUIDs)
    vis_metadata = {
        "filtered_ids": filtered_ids,               # List[str], len=n_kept
        "filtered_vis_count": filtered_vis_count,    # np.ndarray [n_kept]
        "filtered_vis_frames": filtered_vis_frames,  # List[List[int]], len=n_kept
        "per_frame_visible_ids": per_frame_visible_ids,  # {frame_idx: set(uuid)}
        "per_frame_boxes_by_uuid": per_frame_boxes_by_uuid,  # {frame_idx: {uuid: [8,3]}}
    }
    return batch, vis_metadata


def export_predictions(step, predictions, batch, output_dir,
                       per_frame_gt=None, vis_metadata=None):
    """Export predictions to .pkl for offline visualization.

    Args:
        per_frame_gt: list of per-frame GT dicts from bbox_corners.
        vis_metadata: dict from load_fixed_batch() with UUID-level visibility info.
    """
    os.makedirs(output_dir, exist_ok=True)

    logits = predictions["qcod_logits"][0].detach().float().cpu()  # [O, 2]
    scores = F.softmax(logits, dim=-1)[:, 1]  # foreground probability

    data = {
        "seq_name": SEQ_NAME,
        "image_idxs": IMAGE_IDXS,
        "first_frame_idx": IMAGE_IDXS[0],
        "step": step,
        # ── Predictions (in sampled-first-frame camera + normalized coords) ──
        "pred_corners_norm": predictions["qcod_corners"][0].detach().float().cpu().numpy(),
        "pred_center_norm": predictions["qcod_center"][0].detach().float().cpu().numpy(),
        "pred_size_norm": predictions["qcod_size"][0].detach().float().cpu().numpy(),
        "pred_R": predictions["qcod_R"][0].detach().float().cpu().numpy(),
        "pred_scores": scores.numpy(),
        "pred_logits_raw": logits.numpy(),
        # ── Scene GT (vis-filtered, in first-frame camera + normalized) ──
        "gt_scene_corners_norm": batch["qcod_scene_corners"][0][batch["qcod_scene_valid"][0]].cpu().numpy(),
        "gt_scene_R": batch["qcod_scene_R"][0][batch["qcod_scene_valid"][0]].cpu().numpy(),
        "gt_scene_scale": batch["qcod_scene_scale"][0][batch["qcod_scene_valid"][0]].cpu().numpy(),
        "gt_scene_num_instances": int(batch["qcod_scene_valid"][0].sum().item()),
        # ── Per-frame visible GT (filtered to the 4 chosen frames, per-frame cam coords) ──
        "gt_per_frame": per_frame_gt if per_frame_gt is not None else [],
        # ── Camera params ──
        "intrinsics": batch["intrinsics"][0].cpu().numpy(),
        "extrinsics_normalized": batch["extrinsics"][0].cpu().numpy(),
    }

    # Add UUID visibility metadata if available
    if vis_metadata is not None:
        data["vis_filtered_ids"] = vis_metadata["filtered_ids"]
        data["vis_count_per_instance"] = vis_metadata["filtered_vis_count"].tolist()
        data["vis_frames_per_instance"] = vis_metadata["filtered_vis_frames"]
        # Per-frame boxes keyed by UUID (raw camera coords, NOT normalized)
        data["per_frame_boxes_by_uuid"] = {
            int(f): {uid: box.tolist() for uid, box in boxes.items()}
            for f, boxes in vis_metadata["per_frame_boxes_by_uuid"].items()
        }

    path = os.path.join(output_dir, f"step_{step:06d}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def main():
    print("=" * 60)
    print(f"QCOD Single-Scene Overfit: {SEQ_NAME}")
    print(f"Frames: {IMAGE_IDXS}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # ── Build model ──
    print("\n1. Building model...")
    from vggt.models.vggt import VGGT
    from train_utils.freeze import freeze_modules

    model = VGGT(
        enable_camera=True,
        enable_gravity=True,
        enable_depth=True,
        enable_point=False,
        enable_track=False,
        enable_cubify=False,
        enable_qcod=True,
        num_object_queries=256, 
        qcod_query_dim=512,
        qcod_fada_layers="none",
    ).to(DEVICE)

    # Load pretrained weights for frozen modules
    ckpt_candidates = [
        "/home/fanwg/model/model.pt"
        # "/data1/huang/pre_weight/vggt/model.pt",
        # "/data1/gaozhirui/models/vggt/model.pt",
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if os.path.exists(p):
            ckpt_path = p
            break

    if ckpt_path is not None:
        print(f"   Loading pretrained weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f"   Missing keys: {len(missing)} (expected: QCOD modules)")
    else:
        print("   WARNING: no pretrained checkpoint found, using random init for frozen modules")

    model = freeze_modules(model, [
        "*aggregator*", "*camera_head*", "*depth_head*", "*gravity_head*"
    ])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable: {trainable / 1e6:.1f}M")

    # ── Build loss ──
    from loss import MultitaskLoss
    loss_fn = MultitaskLoss(
        camera={"weight": 0.0, "loss_type": "l1"},  # camera still computed but weighted 0 for overfit
        depth={"weight": 0.0, "gradient_loss_fn": "grad", "valid_range": 0.98},
        box=None, point=None, track=None,
        qcod={
            "weight": 1.0,
            "loss_weights": {
                "chamfer": 1.0, "center_l1": 3.0, "cls": 1.0,  # center
                "rotation": 1.0, "size": 0.3,
                "activation": 0.0, "fada": 0.0,
            },
            "fada_layers": "none",
        },
    )

    # ── Build optimizer (QCOD params only) ──
    qcod_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(qcod_params, lr=LR, weight_decay=0.01)

    # ── Load fixed batch ──
    print("\n2. Loading fixed batch...")
    batch, vis_metadata = load_fixed_batch()
    # Move to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(DEVICE)
    n_gt = int(batch["qcod_scene_valid"][0].sum().item())
    print(f"   images: {batch['images'].shape}, filtered GT instances: {n_gt}")

    # ── Build per-frame visible GT (from batch['bbox_corners'] for backward compat) ──
    bbox_per_batch = batch["bbox_corners"][0]  # [S*500, 8, 3]
    S = batch["images"].shape[1]
    bbox_reshaped = bbox_per_batch.view(S, 500, 8, 3)
    per_frame_gt = []
    for s, img_idx in enumerate(IMAGE_IDXS):
        frame_corners = bbox_reshaped[s]
        valid = frame_corners.abs().sum(dim=[-2, -1]) > 1e-6
        visible_corners = frame_corners[valid].cpu().numpy()
        per_frame_gt.append({
            "image_idx": img_idx,
            "frame_index_in_seq": s,
            "corners_cam_norm": visible_corners,
            "num_visible": int(valid.sum().item()),
        })
        print(f"   frame {img_idx} (seq idx {s}): {per_frame_gt[-1]['num_visible']} visible boxes")

    # ── Training loop ──
    print(f"\n3. Overfit training ({NUM_STEPS} steps, lr={LR})...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.train()

    # Export step 0 (before any training)
    with torch.no_grad(), torch.amp.autocast(DEVICE.type, dtype=torch.bfloat16):
        preds_init = model(batch["images"])
    path = export_predictions(0, preds_init, batch, OUTPUT_DIR,
                              per_frame_gt=per_frame_gt, vis_metadata=vis_metadata)
    print(f"   [step 0] exported to {path}")

    scaler = torch.amp.GradScaler()

    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()

        with torch.amp.autocast(DEVICE.type, dtype=torch.bfloat16):
            predictions = model(batch["images"])
            loss_dict = loss_fn(predictions, batch)

        total_loss = loss_dict["objective"]
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(qcod_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % LOG_EVERY == 0 or step == 1:
            qcod_total = loss_dict.get("loss_qcod_total", torch.tensor(0.0)).item()
            chamfer = loss_dict.get("loss_qcod_chamfer", torch.tensor(0.0)).item()
            center = loss_dict.get("loss_qcod_center", torch.tensor(0.0)).item()
            cls_val = loss_dict.get("loss_qcod_cls", torch.tensor(0.0)).item()
            rot = loss_dict.get("loss_qcod_rotation", torch.tensor(0.0)).item()
            size = loss_dict.get("loss_qcod_size", torch.tensor(0.0)).item()
            matched = loss_dict.get("qcod_num_matched", 0)
            # Diagnostic: per-query diversity in predictions (anti-collapse check)
            with torch.no_grad():
                pc = predictions["qcod_center"][0].float()  # [O, 3]
                pred_std = pc.std(dim=0).mean().item()
                pred_range = (pc.max(dim=0).values - pc.min(dim=0).values).mean().item()
                anchor_std = model.qcod_head.box_head.query_anchor.std(dim=0).mean().item()
            print(
                f"   [step {step:4d}] total={qcod_total:.4f} "
                f"cham={chamfer:.4f} cen={center:.4f} cls={cls_val:.4f} "
                f"rot={rot:.4f} size={size:.4f} m={matched} "
                f"| pred_std={pred_std:.4f} range={pred_range:.4f} anch_std={anchor_std:.4f}"
            )

        if step % EXPORT_EVERY == 0:
            with torch.no_grad():
                path = export_predictions(step, predictions, batch, OUTPUT_DIR,
                                          per_frame_gt=per_frame_gt, vis_metadata=vis_metadata)
            print(f"   → exported to {path}")

    # Final export
    with torch.no_grad(), torch.amp.autocast(DEVICE.type, dtype=torch.bfloat16):
        final_preds = model(batch["images"])
    path = export_predictions(NUM_STEPS, final_preds, batch, OUTPUT_DIR, per_frame_gt=per_frame_gt)
    print(f"\n   Final export: {path}")

    if DEVICE.type == "cuda":
        print(f"\n   Peak GPU memory: {torch.cuda.max_memory_allocated(DEVICE) / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print(f"Overfit training complete. Outputs in {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
