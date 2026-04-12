"""Stage A Integration Test — covers the Phase 3 gap.

Validates the FULL chain that Phase 3 smoke test didn't cover:
    VGGT(enable_qcod=True).forward(images)
    → predictions with qcod_* keys
    → MultitaskLoss(qcod=Stage_A_cfg).forward(predictions, batch)
    → real compute_qcod_loss (Hungarian matching, 6 loss terms)
    → backward through the ENTIRE pipeline

This uses REAL batch data (CA1M) + REAL model forward + REAL loss, on GPU.

Run with:
    CUDA_VISIBLE_DEVICES=1 conda run -n superbox python tests/test_qcod_stage_a_integration.py
"""

import os
import sys
import types

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
from torch.utils.data.dataloader import default_collate
from omegaconf import OmegaConf


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


def make_real_batch(device, num_samples=1):
    """Load a real batch and normalize it (same as Phase 1/2 tests)."""
    from data.composed_dataset import ComposedDataset
    from trainer import Trainer

    cfg = OmegaConf.create({
        "dataset_configs": [{
            "_target_": "data.datasets.ca1m.CA1MDataset",
            "split": "train",
            "CA1M_DIR": "/data1/lyq/CA1M-dataset/CA1M-dataset/training/",
            "CA1M_ANNOTATION_DIR": "/data1/lyq/CA1M-dataset/CA1M-dataset/",
        }],
    })
    ds = ComposedDataset(dataset_configs=cfg.dataset_configs,
                         common_config=make_common_config())
    samples = [ds[(i, 4, 1.0)] for i in range(num_samples)]
    cleaned = [{k: v for k, v in s.items() if v is not None} for s in samples]
    batch = default_collate(cleaned)

    # Normalize via Trainer._process_batch (mock self)
    from types import SimpleNamespace
    fake_self = SimpleNamespace(
        data_conf=SimpleNamespace(
            train=SimpleNamespace(
                common_config=SimpleNamespace(repeat_batch=False)
            )
        )
    )
    batch = Trainer._process_batch(fake_self, batch)

    # Move to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def main():
    print("=" * 60)
    print("Stage A Integration Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Build model ──
    print("\n1. Building VGGT(enable_qcod=True)...")
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
        num_object_queries=64,
        qcod_query_dim=512,
        qcod_fada_layers="none",
    ).to(device)

    model = freeze_modules(model, [
        "*aggregator*", "*camera_head*", "*depth_head*", "*gravity_head*"
    ])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable: {trainable / 1e6:.1f}M")

    # ── Build MultitaskLoss with QCOD config ──
    print("\n2. Building MultitaskLoss(qcod=Stage_A)...")
    from loss import MultitaskLoss

    loss_fn = MultitaskLoss(
        camera={"weight": 5.0, "loss_type": "l1"},
        depth={"weight": 1.0, "gradient_loss_fn": "grad", "valid_range": 0.98},
        box=None,
        point=None,
        track=None,
        qcod={
            "weight": 1.0,
            "loss_weights": {
                "chamfer": 1.0, "center_l1": 3.0, "cls": 1.0,
                "rotation": 1.0, "size": 0.3,
                "activation": 0.0, "fada": 0.0,
            },
            "fada_layers": "none",
        },
    )
    print("   MultitaskLoss built OK")

    # ── Load real batch ──
    print("\n3. Loading real batch...")
    batch = make_real_batch(device, num_samples=1)
    print(f"   images: {batch['images'].shape}")
    valid_count = batch["qcod_scene_valid"].sum().item()
    print(f"   qcod_scene_valid: {valid_count} instances")

    # ── Forward pass (full model) ──
    print("\n4. Forward pass (VGGT + QCOD)...")
    model.train()
    with torch.amp.autocast(device.type, dtype=torch.bfloat16):
        predictions = model(batch["images"])

    # Verify QCOD keys present
    assert "qcod_corners" in predictions, "qcod_corners missing from predictions"
    assert "qcod_cross_attn_weights" in predictions
    assert len(predictions["qcod_cross_attn_weights"]) == 0, "Stage A should have no attn weights"
    print(f"   predictions OK: qcod_corners {predictions['qcod_corners'].shape}")

    # ── MultitaskLoss forward (the gap from Phase 3) ──
    print("\n5. MultitaskLoss forward (REAL compute_qcod_loss path)...")
    with torch.amp.autocast(device.type, dtype=torch.bfloat16):
        loss_dict = loss_fn(predictions, batch)

    print(f"   objective: {loss_dict['objective'].item():.4f}")

    # Check QCOD loss keys exist
    assert "loss_qcod_total" in loss_dict, "loss_qcod_total missing"
    assert "loss_qcod_chamfer" in loss_dict
    assert "loss_qcod_center" in loss_dict
    assert "loss_qcod_cls" in loss_dict
    assert "loss_qcod_rotation" in loss_dict
    assert "loss_qcod_size" in loss_dict
    assert "loss_qcod_activation" in loss_dict
    assert "loss_qcod_fada" in loss_dict

    # Stage A: activation and fada should be 0
    assert loss_dict["loss_qcod_activation"].item() == 0.0
    assert loss_dict["loss_qcod_fada"].item() == 0.0

    print(f"   qcod_chamfer:  {loss_dict['loss_qcod_chamfer'].item():.4f}")
    print(f"   qcod_center:   {loss_dict['loss_qcod_center'].item():.4f}")
    print(f"   qcod_cls:      {loss_dict['loss_qcod_cls'].item():.4f}")
    print(f"   qcod_rotation: {loss_dict['loss_qcod_rotation'].item():.4f}")
    print(f"   qcod_size:     {loss_dict['loss_qcod_size'].item():.4f}")
    print(f"   qcod_total:    {loss_dict['loss_qcod_total'].item():.4f}")
    print(f"   qcod_matched:  {loss_dict['qcod_num_matched']}")

    assert loss_dict["qcod_num_matched"] > 0, "No Hungarian matches — data or pred issue"
    assert loss_dict["loss_qcod_total"].item() > 0, "Total QCOD loss should be > 0"

    # ── Backward ──
    print("\n6. Backward through MultitaskLoss...")
    loss_dict["objective"].backward()

    # Verify gradients reach QCOD modules
    qe_grad_count = sum(
        1 for p in model.query_evolution.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    qh_grad_count = sum(
        1 for p in model.qcod_head.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"   query_evolution params with grad: {qe_grad_count}")
    print(f"   qcod_head params with grad: {qh_grad_count}")
    assert qe_grad_count > 0, "No gradients reached query_evolution via MultitaskLoss"
    assert qh_grad_count > 0, "No gradients reached qcod_head via MultitaskLoss"

    # Aggregator should have no gradient
    agg_grads = sum(
        1 for p in model.aggregator.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert agg_grads == 0, f"Aggregator has {agg_grads} params with gradient (should be frozen)"

    # ── Memory check ──
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\n7. Peak GPU memory: {peak_mem:.2f} GB")
    else:
        print("\n7. (CPU mode, no GPU memory check)")

    print("\n" + "=" * 60)
    print("Stage A Integration Test: ALL CHECKS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
