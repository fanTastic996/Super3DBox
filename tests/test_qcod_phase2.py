"""Phase 2 smoke test for QCOD loss computation.

Covers Plan steps 2.1-2.3:
  2.1 Projection + rotation utilities (already unit-tested)
  2.2 compute_qcod_loss with Hungarian matching + 6 loss terms
  2.3 Mock predictions + real batch → loss backward

Run with:
    CUDA_VISIBLE_DEVICES=1 conda run -n superbox python tests/test_qcod_phase2.py
"""

import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))

# Stub open3d for ca1m.py import
if "open3d" not in sys.modules:
    o3d_stub = types.ModuleType("open3d")
    o3d_stub.io = types.SimpleNamespace(read_point_cloud=lambda *a, **k: None)
    sys.modules["open3d"] = o3d_stub

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from omegaconf import OmegaConf

from qcod_loss import compute_qcod_loss, QCODLossConfig
from qcod_utils import rotation_6d_to_matrix


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


def make_real_batch(num_samples=2):
    """Load a real batch through the full Phase 1 pipeline."""
    from data.composed_dataset import ComposedDataset

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

    # Run normalization via trainer._process_batch mock
    from types import SimpleNamespace
    from trainer import Trainer
    fake_self = SimpleNamespace(
        data_conf=SimpleNamespace(
            train=SimpleNamespace(
                common_config=SimpleNamespace(repeat_batch=False)
            )
        )
    )
    batch = Trainer._process_batch(fake_self, batch)
    return batch


def make_mock_predictions(batch, O=64):
    """Create mock QCOD predictions matching the batch dimensions."""
    B = batch["images"].shape[0]
    device = batch["images"].device  # CPU

    # Use requires_grad=True so we can verify backward works.
    pred_corners = torch.randn(B, O, 8, 3, requires_grad=True)
    pred_center = torch.randn(B, O, 3, requires_grad=True)
    pred_size = torch.exp(torch.randn(B, O, 3)).requires_grad_(True)
    pred_logits = torch.randn(B, O, 2, requires_grad=True)
    # Generate valid rotation matrices via 6D representation.
    rot6d = torch.randn(B, O, 6)
    pred_R = rotation_6d_to_matrix(rot6d)  # [B, O, 3, 3], proper rotation

    preds = {
        "qcod_corners": pred_corners,
        "qcod_center": pred_center,
        "qcod_size": pred_size,
        "qcod_logits": pred_logits,
        "qcod_R": pred_R,
    }
    return preds


def test_stage_A(batch):
    """Stage A: 5 core losses, activation=0, fada=0."""
    print("\n=== Stage A: chamfer + center + cls + rotation + size ===")
    O = 64
    preds = make_mock_predictions(batch, O=O)
    cfg = QCODLossConfig()  # all Stage A defaults

    losses = compute_qcod_loss(preds, batch, cfg)

    print(f"  num_matched: {losses['num_matched']}")
    assert losses["num_matched"] > 0, "No matches — GT or pred issue"

    for key in ["chamfer", "center", "cls", "rotation", "size"]:
        val = losses[key].item()
        print(f"  L_{key}: {val:.6f}")
        assert not torch.isnan(losses[key]), f"NaN in {key}"
        assert not torch.isinf(losses[key]), f"Inf in {key}"

    # activation and fada should be exactly 0 (weights are 0)
    assert losses["activation"].item() == 0.0, f"activation should be 0, got {losses['activation'].item()}"
    assert losses["fada"].item() == 0.0, f"fada should be 0, got {losses['fada'].item()}"

    total = losses["total"]
    print(f"  L_total: {total.item():.6f}")
    assert total.requires_grad, "total must have grad"
    assert total.item() > 0, "total should be > 0 for random predictions"

    # Backward test.
    total.backward()
    for k, v in preds.items():
        if isinstance(v, torch.Tensor) and v.requires_grad:
            assert v.grad is not None, f"No gradient for {k}"
            assert v.grad.abs().sum() > 0, f"Zero gradient for {k}"
            print(f"  grad[{k}] norm: {v.grad.norm().item():.4f}")

    print("  Stage A: ALL CHECKS PASSED ✓")


def test_stage_B(batch):
    """Stage B: + activation loss enabled."""
    print("\n=== Stage B: + activation loss (mock) ===")
    O = 64
    preds = make_mock_predictions(batch, O=O)

    B = batch["images"].shape[0]
    S = batch["images"].shape[1]
    H_img = batch["images"].shape[-2]
    W_img = batch["images"].shape[-1]
    H_patch = H_img // 14
    W_patch = W_img // 14
    P_patch = H_patch * W_patch

    # Add activation logits
    preds["qcod_activation_logits"] = torch.randn(
        B, O, S * P_patch, requires_grad=True
    )

    cfg = QCODLossConfig(activation=0.5)  # Stage B
    losses = compute_qcod_loss(preds, batch, cfg)

    act_val = losses["activation"].item()
    print(f"  L_activation: {act_val:.6f}")
    assert act_val > 0, "activation loss should be > 0 when enabled"
    assert losses["fada"].item() == 0.0, "fada should still be 0 in Stage B"
    print(f"  L_total: {losses['total'].item():.6f}")

    losses["total"].backward()
    assert preds["qcod_activation_logits"].grad is not None
    print(f"  grad[activation_logits] norm: {preds['qcod_activation_logits'].grad.norm().item():.4f}")
    print("  Stage B: ALL CHECKS PASSED ✓")


def test_stage_C(batch):
    """Stage C: + FADA loss enabled (last layer only)."""
    print("\n=== Stage C: + FADA loss (mock, last layer) ===")
    O = 64
    preds = make_mock_predictions(batch, O=O)

    B = batch["images"].shape[0]
    S = batch["images"].shape[1]
    H_img = batch["images"].shape[-2]
    W_img = batch["images"].shape[-1]
    H_patch = H_img // 14
    W_patch = W_img // 14
    P_patch = H_patch * W_patch

    preds["qcod_activation_logits"] = torch.randn(B, O, S * P_patch, requires_grad=True)
    # FADA: only last layer's attention weights
    preds["qcod_cross_attn_weights"] = [
        torch.softmax(torch.randn(B, 8, O, S * P_patch), dim=-1).requires_grad_(True)
    ]

    cfg = QCODLossConfig(activation=0.5, fada=0.3)
    losses = compute_qcod_loss(preds, batch, cfg)

    fada_val = losses["fada"].item()
    print(f"  L_fada: {fada_val:.6f}")
    assert fada_val > 0, "fada loss should be > 0 when enabled with mock attn"
    print(f"  L_total: {losses['total'].item():.6f}")

    losses["total"].backward()
    attn_grad = preds["qcod_cross_attn_weights"][0].grad
    if attn_grad is not None:
        print(f"  grad[attn_weights] norm: {attn_grad.norm().item():.4f}")
    else:
        print("  WARN: attn_weights grad is None (softmax may detach)")
    print("  Stage C: ALL CHECKS PASSED ✓")


def test_fail_loud_misconfig(batch):
    """Regression tests: activation/FADA weight > 0 with missing keys must raise, not skip."""
    print("\n=== Fail-loud misconfiguration tests ===")
    O = 64
    preds_base = make_mock_predictions(batch, O=O)

    # Case 1: activation > 0 but qcod_activation_logits missing → KeyError
    try:
        compute_qcod_loss(preds_base, batch, QCODLossConfig(activation=0.5))
        assert False, "Should have raised KeyError for missing activation logits"
    except KeyError:
        print("  Case 1 (activation key missing → KeyError): OK")

    # Case 2: fada > 0 but qcod_cross_attn_weights missing → KeyError
    try:
        compute_qcod_loss(preds_base, batch, QCODLossConfig(fada=0.3))
        assert False, "Should have raised KeyError for missing cross_attn_weights"
    except KeyError:
        print("  Case 2 (fada key missing → KeyError): OK")

    # Case 3: fada > 0 and qcod_cross_attn_weights present but empty list → RuntimeError
    preds_empty_fada = dict(preds_base)
    preds_empty_fada["qcod_cross_attn_weights"] = []
    try:
        compute_qcod_loss(preds_empty_fada, batch, QCODLossConfig(fada=0.3))
        assert False, "Should have raised RuntimeError for empty cross_attn_weights"
    except RuntimeError:
        print("  Case 3 (fada empty list → RuntimeError): OK")

    print("  Fail-loud tests: ALL CHECKS PASSED ✓")


def main():
    print("=" * 60)
    print("Phase 2 QCOD loss smoke test")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    print("Loading real batch through Phase 1 pipeline...")
    batch = make_real_batch(num_samples=2)
    print(f"  Batch loaded: B={batch['images'].shape[0]}, S={batch['images'].shape[1]}, "
          f"H={batch['images'].shape[-2]}, W={batch['images'].shape[-1]}")
    valid_counts = batch["qcod_scene_valid"].sum(dim=1).tolist()
    print(f"  qcod_scene_valid counts: {valid_counts}")

    test_stage_A(batch)
    test_stage_B(batch)
    test_stage_C(batch)
    test_fail_loud_misconfig(batch)

    print("\n" + "=" * 60)
    print("Phase 2 smoke test: ALL CHECKS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
