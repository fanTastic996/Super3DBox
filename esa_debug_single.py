"""
Fast single-scene ESA debug with full loss pipeline.

Losses:
  L_mask  = 5.0 * Dice + 5.0 * Focal (attention map vs projected GT mask)
  L_box   = Chamfer(pred_corners, gt_corners) + 3.0 * L1(center)
  L_obj   = Hard Negative Mining BCE (slot-level fg/bg)
  L_total = 2.0 * L_mask + 1.0 * L_box + 1.0 * L_obj

Temperature: anneal from 1.0 → 0.5 over first 20 epochs.
Output: .pkl with predictions for offline evaluation.
"""

import os, sys, glob, time, pickle, numpy as np
import torch
import torch.nn.functional as F

VGGT_ROOT = "/home/fanwg/myproject/code/vggt"
sys.path.insert(0, VGGT_ROOT)
sys.path.insert(0, os.path.join(VGGT_ROOT, "training"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from enhanced_slot_attention import (
    ESAValidationModel, CornersBoxHead, SlotClassifier,
    select_covisible_cameras, generate_gt_attention_masks,
    normalize_gt_corners,
    chamfer_loss, compute_box_matching_cost, compute_mask_matching_cost,
    hungarian_matching,
    attention_mask_loss, hardneg_classification_loss,
    project_boxes_to_objectness,
)

CA1M_DIR = "/data1/lyq/CA1M-dataset/CA1M-dataset/test"
MODEL_CKPT = "/home/lanyuqing/model/model.pt"
IMG_SIZE, PATCH_SIZE = 518, 14
GRID_H = GRID_W = IMG_SIZE // PATCH_SIZE
N_PATCHES = GRID_H * GRID_W

NUM_SLOTS = 64
NUM_GLOBAL = 2
SLOT_DIM = 256
NUM_ITER = 3
EPOCHS = 500
LR = 5e-4
TEMP_START, TEMP_END, TEMP_ANNEAL_EPOCHS = 1.0, 1.0, 20  # fixed at 1.0 until GT mask verified

VIS_DIR = os.path.join(VGGT_ROOT, "vis_debug_single")
PKL_DIR = os.path.join(VGGT_ROOT, "pkl_output")


def find_good_scene(max_scan=300):
    """Find a scene with good co-visibility and multiple objects."""
    scene_dirs = sorted(glob.glob(os.path.join(CA1M_DIR, "*")))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
    np.random.seed(42)
    np.random.shuffle(scene_dirs)

    for sd in scene_dirs[:max_scan]:
        K_file = os.path.join(sd, "K_rgb.txt")
        poses_file = os.path.join(sd, "all_poses.npy")
        corners_file = os.path.join(sd, "after_filter_boxes.npy")
        if not all(os.path.exists(f) for f in [K_file, poses_file, corners_file]):
            continue

        K = np.loadtxt(K_file).reshape(3, 3).astype(np.float32)
        poses = np.load(poses_file).astype(np.float32)
        boxes = np.load(corners_file).astype(np.float32)
        if len(boxes) < 3 or len(poses) < 4:
            continue

        oh, ow = int(2 * K[1, 2]), int(2 * K[0, 2])
        sel = select_covisible_cameras(boxes, poses, K, oh, ow, num_cameras=4)
        if sel is None:
            continue

        total_pos = 0
        for idx in sel:
            labels = project_boxes_to_objectness(boxes, K, poses[idx], oh, ow, GRID_H, GRID_W)
            total_pos += labels.sum()

        pos_ratio = total_pos / (4 * N_PATCHES)
        if pos_ratio > 0.03:
            log.info(f"Scene: {os.path.basename(sd)}, {len(boxes)} boxes, pos={pos_ratio:.3f}")
            return sd, sel, K, poses, boxes, oh, ow

    return None


@torch.no_grad()
def extract_single_scene(model, scene_dir, sel_indices):
    from vggt.utils.load_fn import load_and_preprocess_images_square

    rgb_dir = os.path.join(scene_dir, "rgb")

    # Build paths directly from frame IDs to avoid lexicographic sort issues
    selected_paths = []
    for idx in sel_indices:
        p = os.path.join(rgb_dir, f"{idx}.png")
        if not os.path.exists(p):
            p = os.path.join(rgb_dir, f"{idx}.jpg")
        selected_paths.append(p)

    images, _ = load_and_preprocess_images_square(selected_paths, IMG_SIZE)
    images_gpu = images.unsqueeze(0).to("cuda")

    agg_list, patch_start = model.aggregator(images_gpu)
    last = agg_list[-1]
    features = last[:, :, patch_start:, :].squeeze(0).cpu().half()

    del images_gpu, agg_list, last
    torch.cuda.empty_cache()
    return features, selected_paths


def get_temperature(epoch):
    """Anneal temperature from TEMP_START to TEMP_END over TEMP_ANNEAL_EPOCHS."""
    if epoch >= TEMP_ANNEAL_EPOCHS:
        return TEMP_END
    alpha = epoch / TEMP_ANNEAL_EPOCHS
    return TEMP_START + (TEMP_END - TEMP_START) * alpha


def train_debug(features, gt_corners_norm, gt_masks_t, gt_visible,
                all_poses, sel_indices, epochs=EPOCHS):
    """Full training loop with L_mask + L_box + L_obj."""
    device = "cuda"
    S = features.shape[0]

    # Models
    esa = ESAValidationModel(
        in_dim=2048, slot_dim=SLOT_DIM, num_slots=NUM_SLOTS,
        num_global_slots=NUM_GLOBAL, num_iterations=NUM_ITER, max_views=8,
    ).to(device)
    box_head = CornersBoxHead(dim=SLOT_DIM).to(device)
    classifier = SlotClassifier(dim=SLOT_DIM).to(device)

    # Separate param groups: logit_scale and sink_bias get 10x LR
    scale_params = {id(esa.slot_attention.logit_scale), id(esa.slot_attention.sink_bias)}
    base_params = []
    fast_params = []
    for p in list(esa.parameters()) + list(box_head.parameters()) + list(classifier.parameters()):
        if id(p) in scale_params:
            fast_params.append(p)
        else:
            base_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": LR},
        {"params": fast_params, "lr": LR * 10},
    ], weight_decay=1e-4)
    all_params = base_params + fast_params

    feat_gpu = features.float().unsqueeze(0).to(device)
    gt_corners_t = torch.from_numpy(gt_corners_norm).to(device)  # [N_gt, 8, 3]

    # Filter to only visible GT instances
    vis_idx = np.where(gt_visible)[0]
    if len(vis_idx) == 0:
        log.error("No visible GT instances!")
        return esa, box_head, classifier

    gt_corners_vis = gt_corners_t[vis_idx]   # [M, 8, 3]
    gt_masks_vis = gt_masks_t[vis_idx]       # [M, N]

    log.info(f"Visible GT instances: {len(vis_idx)} / {len(gt_corners_norm)}")
    log.info(f"GT masks sum: {gt_masks_vis.sum(dim=-1).cpu().numpy()}")

    best_loss = float("inf")
    best_state = {}

    for epoch in range(epochs):
        esa.train(); box_head.train(); classifier.train()
        optimizer.zero_grad()

        temp = get_temperature(epoch)

        # Forward ESA (returns K+1 slots: K foreground + 1 sink)
        slots_all, attn_all, all_attns_all = esa(feat_gpu, S, temperature=temp)
        # slots_all: [1, K+1, D], attn_all: [1, K+1, N]

        K = NUM_SLOTS  # foreground slot count (exclude sink)

        # Separate foreground slots from sink
        slots_fg = slots_all[:, :K, :]       # [1, K, D]
        attn_fg = attn_all[:, :K, :]         # [1, K, N]
        attn_sink = attn_all[:, K:, :]       # [1, 1, N]

        # Box + classification predictions (foreground only)
        pred_corners = box_head(slots_fg)     # [1, K, 8, 3]
        pred_logits = classifier(slots_fg)    # [1, K, 2]

        # Squeeze batch dim (single scene)
        pred_c = pred_corners.squeeze(0)      # [K, 8, 3]
        pred_l = pred_logits.squeeze(0)       # [K, 2]
        attn_s = attn_fg.squeeze(0)           # [K, N]

        # ── Hungarian matching (mask-based, foreground only) ──
        cost = compute_mask_matching_cost(attn_s, gt_masks_vis, pred_l)
        row_ind, col_ind = hungarian_matching(cost)

        # ── L_box: Chamfer + 3.0 * Center L1 ──
        matched_pred = pred_c[row_ind]        # [P, 8, 3]
        matched_gt = gt_corners_vis[col_ind]  # [P, 8, 3]

        l_chamfer = chamfer_loss(matched_pred, matched_gt)
        pred_center = matched_pred.mean(dim=1)  # [P, 3]
        gt_center = matched_gt.mean(dim=1)      # [P, 3]
        l_center = F.l1_loss(pred_center, gt_center)
        l_box = l_chamfer + 3.0 * l_center

        # ── L_mask: 5*Dice + 5*Focal, graduated over all iterations ──
        # Foreground slots only (exclude sink from mask loss)
        gamma = 0.5
        iter_weights = [gamma ** (NUM_ITER - 1 - t) for t in range(NUM_ITER)]
        l_mask = torch.tensor(0.0, device=device)
        for t, (a_t, w_t) in enumerate(zip(all_attns_all, iter_weights)):
            a_t_fg = a_t.squeeze(0)[:K, :]  # [K, N] foreground only
            l_mask = l_mask + w_t * attention_mask_loss(
                a_t_fg, gt_masks_vis, row_ind, col_ind
            )
        l_mask = l_mask / sum(iter_weights)

        # ── L_obj: Hard Negative Mining (foreground only) ──
        l_obj = hardneg_classification_loss(pred_l, row_ind, num_hard_neg=5)

        # ── L_sink: weak capacity constraint on sink slot ──
        # L_sink = λ * max(0, mass_sink - τ), only fires if sink is too greedy
        sink_mass = attn_sink.sum(dim=-1).mean()  # fraction of total attention
        sink_mass_ratio = sink_mass / (attn_all.sum(dim=-1).mean() + 1e-8)
        l_sink = 0.0 * torch.clamp(sink_mass_ratio - 0.5, min=0.0) # 先破除sink_slot的正则化约束

        # ── Total: 2:1:1 + sink ──
        loss = 2.0 * l_mask + 1.0 * l_box + 1.0 * l_obj + l_sink

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                "esa": {k: v.cpu().clone() for k, v in esa.state_dict().items()},
                "box": {k: v.cpu().clone() for k, v in box_head.state_dict().items()},
                "cls": {k: v.cpu().clone() for k, v in classifier.state_dict().items()},
            }

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                a_safe = attn_s.detach().clamp(1e-10).cpu().numpy()
                entropy = -(a_safe * np.log(a_safe)).sum(axis=0).mean()
                fg_probs = F.softmax(pred_l.detach(), dim=-1)[:, 1].cpu().numpy()
                cur_scale = torch.clamp(esa.slot_attention.logit_scale, 0.1, 10.0).item()
                sink_r = sink_mass_ratio.item()

            log.info(
                f"Ep {epoch+1:3d} | L={loss.item():.4f} "
                f"(mask={l_mask.item():.3f} box={l_box.item():.3f} "
                f"obj={l_obj.item():.3f} sink={l_sink.item():.4f}) | "
                f"scale={cur_scale:.2f} ent={entropy:.2f} sink%={sink_r:.2f} "
                f"fg_top5={np.sort(fg_probs)[-5:][::-1].tolist()}"
            )

    # Load best
    if best_state:
        esa.load_state_dict(best_state["esa"])
        box_head.load_state_dict(best_state["box"])
        classifier.load_state_dict(best_state["cls"])

    return esa, box_head, classifier


def save_predictions_pkl(esa, box_head, classifier, features, gt_corners_norm,
                         gt_masks_t, gt_visible, scene_dir, sel_indices, S):
    """Save predictions as .pkl for offline evaluation."""
    device = "cuda"
    esa.eval(); box_head.eval(); classifier.eval()

    K = NUM_SLOTS  # foreground slots only

    with torch.no_grad():
        feat_gpu = features.float().unsqueeze(0).to(device)
        slots_all, attn_all, _ = esa(feat_gpu, S, temperature=TEMP_END)
        slots_fg = slots_all[:, :K, :]
        attn_fg = attn_all[:, :K, :]
        pred_corners = box_head(slots_fg)
        pred_logits = classifier(slots_fg)

    os.makedirs(PKL_DIR, exist_ok=True)

    save_dict = {
        "pred_corners": pred_corners.squeeze(0).cpu().numpy(),   # [K, 8, 3]
        "pred_logits": pred_logits.squeeze(0).cpu().numpy(),     # [K, 2]
        "gt_corners_norm": gt_corners_norm,                      # [N_gt, 8, 3]
        "gt_visible": gt_visible,                                # [N_gt]
        "attn_maps": attn_fg.squeeze(0).cpu().numpy(),           # [K, S*1369]
        "gt_masks": gt_masks_t.cpu().numpy(),                    # [N_gt, S*1369]
        "scene_dir": scene_dir,
        "sel_indices": sel_indices,
        "S": S,
        "num_slots": NUM_SLOTS,
    }

    scene_name = os.path.basename(scene_dir)
    pkl_path = os.path.join(PKL_DIR, f"esa_debug_{scene_name}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(save_dict, f)
    log.info(f"Predictions saved to {pkl_path}")

    # Print summary
    fg_probs = F.softmax(torch.from_numpy(save_dict["pred_logits"]), dim=-1)[:, 1].numpy()
    n_fg = (fg_probs > 0.5).sum()
    log.info(f"  {n_fg} foreground slots (prob > 0.5)")
    log.info(f"  Top-5 fg probs: {np.sort(fg_probs)[-5:][::-1]}")

    return pkl_path


def visualize_debug(esa, box_head, classifier, features, gt_corners_norm,
                    gt_masks_np, gt_visible, img_paths, sel_indices, all_poses,
                    K_mat, oh, ow, save_dir):
    """Visualize slot attention + predicted boxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)
    device = "cuda"
    esa.eval(); box_head.eval(); classifier.eval()

    S = features.shape[0]
    K = NUM_SLOTS  # foreground slots only
    with torch.no_grad():
        feat_gpu = features.float().unsqueeze(0).to(device)
        slots_all, attn_all, all_attns_all = esa(feat_gpu, S, temperature=TEMP_END)
        slots_fg = slots_all[:, :K, :]
        attn_fg = attn_all[:, :K, :]
        all_attns = [a[:, :K, :] for a in all_attns_all]
        pred_corners = box_head(slots_fg).squeeze(0).cpu().numpy()  # [K, 8, 3]
        pred_logits = classifier(slots_fg).squeeze(0)
        fg_probs = F.softmax(pred_logits, dim=-1)[:, 1].cpu().numpy()

    attn_np = attn_fg.squeeze(0).cpu().numpy()  # [K, N]
    mass = attn_np.sum(axis=-1)

    # Top foreground slots by fg probability
    fg_order = np.argsort(-fg_probs)
    top_slots = fg_order[:6]

    orig_imgs = []
    for p in img_paths[:S]:
        img = Image.open(p).convert("RGB").resize((GRID_W * 8, GRID_H * 8))
        orig_imgs.append(np.array(img))

    slot_colors = plt.cm.tab20(np.linspace(0, 1, NUM_SLOTS))

    # ── Heatmaps + GT mask overlay ──
    n_rows = 1 + len(top_slots) + 1  # orig + slots + GT
    fig, axes = plt.subplots(n_rows, S, figsize=(4 * S, 3 * n_rows))

    for v in range(S):
        axes[0, v].imshow(orig_imgs[v])
        axes[0, v].set_title(f"View {v}", fontsize=9)
        axes[0, v].axis("off")

    for rank, si in enumerate(top_slots):
        for v in range(S):
            hm = attn_np[si, v * N_PATCHES:(v + 1) * N_PATCHES].reshape(GRID_H, GRID_W)
            hm_up = np.kron(hm, np.ones((8, 8)))
            axes[rank + 1, v].imshow(orig_imgs[v], alpha=0.4)
            axes[rank + 1, v].imshow(hm_up, cmap="hot", alpha=0.6,
                                      vmin=0, vmax=max(hm.max(), 1e-6))
            if v == 0:
                axes[rank + 1, v].set_ylabel(
                    f"Slot {si}\nfg={fg_probs[si]:.2f}", fontsize=7)
            axes[rank + 1, v].axis("off")

    # GT mask union
    vis_masks = gt_masks_np[gt_visible]
    if len(vis_masks) > 0:
        gt_union = vis_masks.max(axis=0)  # [N]
    else:
        gt_union = np.zeros(S * N_PATCHES)

    for v in range(S):
        gt = gt_union[v * N_PATCHES:(v + 1) * N_PATCHES].reshape(GRID_H, GRID_W)
        gt_up = np.kron(gt, np.ones((8, 8)))
        axes[-1, v].imshow(orig_imgs[v], alpha=0.4)
        axes[-1, v].imshow(gt_up, cmap="Greens", alpha=0.6, vmin=0, vmax=1)
        if v == 0:
            axes[-1, v].set_ylabel("GT Mask", fontsize=8)
        axes[-1, v].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "heatmaps.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Segmentation map ──
    assignment = attn_np.argmax(axis=0).reshape(S, GRID_H, GRID_W)
    fig, axes = plt.subplots(2, S, figsize=(4 * S, 7))
    for v in range(S):
        axes[0, v].imshow(orig_imgs[v])
        axes[0, v].set_title(f"View {v}")
        axes[0, v].axis("off")
        seg_up = np.kron(assignment[v], np.ones((8, 8), dtype=int))
        seg_color = slot_colors[seg_up.flatten()].reshape(GRID_H * 8, GRID_W * 8, 4)
        axes[1, v].imshow(orig_imgs[v], alpha=0.3)
        axes[1, v].imshow(seg_color, alpha=0.7)
        axes[1, v].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "segmap.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Entropy per iteration ──
    fig, ax = plt.subplots(figsize=(6, 4))
    for t, a in enumerate(all_attns):
        a_np = a.squeeze(0).cpu().numpy()
        a_safe = np.clip(a_np, 1e-10, 1.0)
        ent = -(a_safe * np.log(a_safe)).sum(axis=0).mean()
        ax.plot(t + 1, ent, "o", color="coral", markersize=10)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Token Entropy")
    ax.set_title("Attention Sharpening")
    fig.savefig(os.path.join(save_dir, "entropy.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Visualizations saved to {save_dir}")


def main():
    log.info("=" * 50)
    log.info("ESA Single-Scene Debug — Full Loss Pipeline")
    log.info("=" * 50)

    # 1. Load specific scene and frames
    SCENE_ID = "47204573"     
    FRAME_IDS = [101, 409, 444, 505]

    '''
    SCENE_ID = "42898849"
    FRAME_IDS = [186, 361, 569, 828]

    SCENE_ID = "47115525"      大场景
    FRAME_IDS = [144, 156, 164, 238]   

    SCENE_ID = "47204573"      小物体
    FRAME_IDS = [101, 409, 444, 505]

    SCENE_ID = "47333452"      中场景（角落）
    FRAME_IDS = [70, 137, 167, 214]
    '''

    scene_dir = os.path.join(CA1M_DIR, SCENE_ID)
    sel_indices = FRAME_IDS

    K_mat = np.loadtxt(os.path.join(scene_dir, "K_rgb.txt")).reshape(3, 3).astype(np.float32)
    all_poses = np.load(os.path.join(scene_dir, "all_poses.npy")).astype(np.float32)
    bbox_corners = np.load(os.path.join(scene_dir, "after_filter_boxes.npy")).astype(np.float32)

    oh = int(2 * K_mat[1, 2])
    ow = int(2 * K_mat[0, 2])

    log.info(f"Scene: {SCENE_ID}, frames: {FRAME_IDS}")
    log.info(f"  {len(bbox_corners)} boxes, {len(all_poses)} poses, oh={oh}, ow={ow}")

    # 2. Extract features
    log.info("Loading VGGT...")
    from vggt.models.vggt import VGGT
    vggt = VGGT(enable_camera=False, enable_gravity=False, enable_depth=False,
                enable_point=False, enable_track=False, enable_cubify=False)
    ckpt = torch.load(MODEL_CKPT, map_location="cpu")
    vggt.load_state_dict(ckpt, strict=False)
    vggt = vggt.cuda().eval()

    log.info("Extracting features...")
    features, img_paths = extract_single_scene(vggt, scene_dir, sel_indices)
    del vggt; torch.cuda.empty_cache()
    S = features.shape[0]
    log.info(f"Features: {features.shape}")

    # 3. Generate GT
    gt_corners_norm, center, scale = normalize_gt_corners(
        bbox_corners, all_poses[sel_indices[0]]
    )
    gt_masks_np, gt_visible = generate_gt_attention_masks(
        bbox_corners, K_mat, all_poses, sel_indices, oh, ow, GRID_H, GRID_W
    )
    gt_masks_t = torch.from_numpy(gt_masks_np).to("cuda")

    n_vis = gt_visible.sum()
    log.info(f"GT: {len(bbox_corners)} boxes, {n_vis} visible, "
             f"scale={scale:.2f}, center={center}")

    # 4. Train
    log.info(f"\nTraining (epochs={EPOCHS}, K={NUM_SLOTS}, T={NUM_ITER})...")
    t0 = time.time()
    esa, box_head, classifier = train_debug(
        features, gt_corners_norm, gt_masks_t, gt_visible,
        all_poses, sel_indices, epochs=EPOCHS,
    )
    log.info(f"Training done in {time.time()-t0:.1f}s")

    # 5. Save .pkl
    pkl_path = save_predictions_pkl(
        esa, box_head, classifier, features, gt_corners_norm,
        gt_masks_t, gt_visible, scene_dir, sel_indices, S,
    )

    # 6. Visualize
    visualize_debug(
        esa, box_head, classifier, features, gt_corners_norm,
        gt_masks_np, gt_visible, img_paths, sel_indices, all_poses,
        K_mat, oh, ow, VIS_DIR,
    )

    log.info("\nDebug complete!")


if __name__ == "__main__":
    main()