"""Phase 3 end-to-end smoke test for the QCOD head pipeline.

Verifies:
  1. VGGT(enable_qcod=True) instantiates with correct trainable param count
  2. Freeze check: *aggregator* frozen but query_evolution + qcod_head trainable
  3. Forward pass produces qcod_* keys in predictions with correct shapes
  4. Model subgraph backward: gradients flow from a surrogate loss (on corners +
     logits) back through the QCOD pipeline to query_evolution params

NOT verified here (covered by Phase 2 tests instead):
  * MultitaskLoss.forward() with the real QCOD branch (lazy import of
    compute_qcod_loss, Hungarian matching, 6-item loss). That path is tested
    in tests/test_qcod_phase2.py with mock predictions + real batch data.
  * The full VGGT forward → MultitaskLoss → backward chain. This will be
    validated during the Stage A training bring-up.

Run with:
    CUDA_VISIBLE_DEVICES=1 conda run -n superbox python tests/test_qcod_phase3.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))
sys.path.insert(0, REPO_ROOT)

import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


def test_instantiation():
    """Test VGGT with enable_qcod creates the right modules."""
    print("\n=== Test 1: VGGT instantiation with enable_qcod=True ===")
    from vggt.models.vggt import VGGT

    model = VGGT(
        enable_camera=False,
        enable_gravity=False,
        enable_depth=False,
        enable_point=False,
        enable_track=False,
        enable_cubify=False,
        enable_qcod=True,
        num_object_queries=64,
        qcod_query_dim=512,
        qcod_fada_layers="none",
    )

    # Check modules exist
    assert model.query_evolution is not None, "query_evolution missing"
    assert model.qcod_head is not None, "qcod_head missing"
    assert model.query_evolution.num_layers == 24
    assert model.query_evolution.num_queries == 64
    assert model.query_evolution.query_dim == 512
    print("  VGGT instantiation OK")

    return model


def test_freeze(model):
    """Test that *aggregator* freeze doesn't freeze query_evolution / qcod_head."""
    print("\n=== Test 2: Freeze check ===")
    from train_utils.freeze import freeze_modules

    model = freeze_modules(model, ["*aggregator*"])

    frozen = 0
    trainable = 0
    trainable_names = set()
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable += p.numel()
            top_module = name.split(".")[0]
            trainable_names.add(top_module)
        else:
            frozen += p.numel()
            # Make sure aggregator is frozen
            if "aggregator" in name:
                pass  # expected
            # Make sure query_evolution is NOT frozen
            if "query_evolution" in name:
                raise AssertionError(f"query_evolution param frozen: {name}")
            if "qcod_head" in name:
                raise AssertionError(f"qcod_head param frozen: {name}")

    print(f"  Frozen: {frozen / 1e6:.1f}M, Trainable: {trainable / 1e6:.1f}M")
    print(f"  Trainable modules: {trainable_names}")
    assert "query_evolution" in trainable_names
    assert "qcod_head" in trainable_names
    assert trainable > 100e6, f"Expected >100M trainable, got {trainable/1e6:.1f}M"
    print("  Freeze check PASSED")

    return model


def test_forward(model, device):
    """Forward pass produces qcod_* prediction keys."""
    print("\n=== Test 3: Forward pass ===")
    model = model.to(device)
    model.eval()

    B, S = 1, 3
    images = torch.randn(B, S, 3, 518, 518, device=device)

    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        preds = model(images)

    # Check expected keys
    expected_keys = [
        "qcod_corners", "qcod_center", "qcod_size", "qcod_R",
        "qcod_logits", "qcod_activation", "qcod_activation_logits",
        "qcod_cross_attn_weights",
    ]
    for k in expected_keys:
        assert k in preds, f"missing prediction key: {k}"

    # Shapes
    O = 64
    P_patch = (518 // 14) ** 2  # 37 * 37 = 1369
    assert preds["qcod_corners"].shape == (B, O, 8, 3), f"corners: {preds['qcod_corners'].shape}"
    assert preds["qcod_center"].shape == (B, O, 3)
    assert preds["qcod_size"].shape == (B, O, 3)
    assert preds["qcod_R"].shape == (B, O, 3, 3)
    assert preds["qcod_logits"].shape == (B, O, 2)
    assert preds["qcod_activation"].shape == (B, O, S * P_patch)
    assert preds["qcod_activation_logits"].shape == (B, O, S * P_patch)
    # fada_layers="none" → empty list
    assert isinstance(preds["qcod_cross_attn_weights"], list)
    assert len(preds["qcod_cross_attn_weights"]) == 0

    print(f"  predictions OK: {len(expected_keys)} keys, corners {preds['qcod_corners'].shape}")
    return preds


def test_backward(model, device):
    """Model subgraph backward: gradients flow through QCOD to query_evolution.

    Uses a SURROGATE loss (abs-mean of corners + logits), NOT MultitaskLoss.
    This proves the autograd graph is connected from query_evolution → aggregator
    callback → qcod_head → loss, but does NOT test the actual training loss
    pipeline (Hungarian matching, chamfer, etc.). The real loss pipeline is
    covered by tests/test_qcod_phase2.py.
    """
    print("\n=== Test 4: Model subgraph backward (surrogate loss) ===")
    model.train()

    B, S = 1, 3
    images = torch.randn(B, S, 3, 518, 518, device=device)

    with torch.amp.autocast(device.type, dtype=torch.bfloat16):
        preds = model(images)

    # Surrogate loss — NOT MultitaskLoss. Just tests graph connectivity.
    loss = preds["qcod_corners"].float().abs().mean() + preds["qcod_logits"].float().abs().mean()
    loss.backward()

    # Check gradients on query_evolution (at least one param must have nonzero grad)
    qe_grads = 0
    qe_total = 0
    for name, p in model.query_evolution.named_parameters():
        qe_total += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            qe_grads += 1
    print(f"  query_evolution: {qe_grads}/{qe_total} params have non-zero grad")
    assert qe_grads > 0, "No gradients reached query_evolution"

    # Check qcod_head grads (at least one param must have nonzero grad)
    qh_grads = 0
    qh_total = 0
    for name, p in model.qcod_head.named_parameters():
        qh_total += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            qh_grads += 1
    print(f"  qcod_head: {qh_grads}/{qh_total} params have non-zero grad")
    assert qh_grads > 0, "No gradients reached qcod_head"

    # Aggregator should have NO gradient (frozen)
    for name, p in model.aggregator.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            raise AssertionError(f"Aggregator param has gradient: {name}")
    print("  Aggregator has zero grad (frozen) ✓")

    print(f"  Backward PASSED (loss={loss.item():.6f})")


def main():
    print("=" * 60)
    print("Phase 3 QCOD end-to-end smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = test_instantiation()
    model = test_freeze(model)
    test_forward(model, device)
    test_backward(model, device)

    print("\n" + "=" * 60)
    print("Phase 3 smoke test: ALL CHECKS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
