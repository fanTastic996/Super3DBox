"""Phase 1 smoke test for QCOD data pipeline.

Covers Plan steps 1.1-1.5:
  1.1 load_scene_gt_in_first_frame + sanity check (already unit-tested separately)
  1.2 CA1MDataset.get_data() exports qcod_scene_{corners,R,scale,valid}
  1.3 ComposedDataset.__getitem__ passes qcod fields through as torch tensors
  1.4 normalize_camera_extrinsics_and_points_boxes_batch handles new kwargs
  1.5 trainer._process_batch unpacks the new return values and writes back

Verification goals (plan D5 + Phase 1 exit criteria):
  * Single-sample load returns padded numeric tensors, no List[str] fields.
  * default_collate can stack multiple samples into a batch without a custom collate_fn.
  * Normalization preserves corner/scale ratio up to float precision.
  * After normalization, valid corners have reasonable magnitude (< ~20 in avg-scale
    units, typically 1-5 for indoor scenes).

Run with:
    CUDA_VISIBLE_DEVICES=1 conda run -n superbox python tests/test_qcod_phase1.py
"""

import os
import sys
import types

# Make sure training/ is importable (CA1MDataset lives under training/data/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))

# open3d is imported at module load by ca1m.py but only used in load_scene_data()
# which is not on the get_data() path. Stub it so we don't need open3d installed.
if "open3d" not in sys.modules:
    open3d_stub = types.ModuleType("open3d")
    open3d_stub.io = types.SimpleNamespace(read_point_cloud=lambda *a, **k: None)
    sys.modules["open3d"] = open3d_stub

# scipy.spatial.KDTree is also imported at module load; scipy is installed, so this is fine.

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data.dataloader import default_collate


def make_common_config():
    """Minimal common_config matching default_dataset.yaml for CA1M."""
    cfg = OmegaConf.create(
        {
            "fix_img_num": -1,
            "fix_aspect_ratio": 1.0,
            "load_track": False,
            "track_num": 16,
            "training": True,
            "inside_random": False,  # deterministic
            "img_size": 518,
            "patch_size": 14,
            "rescale": True,
            "rescale_aug": False,     # no aug for test determinism
            "landscape_check": False,
            "debug": False,
            "get_nearby": True,
            "load_depth": True,
            "img_nums": [4, 4],       # fixed 4 frames
            "allow_duplicate_img": True,
            "repeat_batch": False,
            "augs": {
                "use": False,
                "cojitter": False,
                "cojitter_ratio": 0.5,
                "scales": None,
                "aspects": [1.0, 1.0],
                "color_jitter": None,
                "gray_scale": False,
                "gau_blur": False,
            },
        }
    )
    return cfg


def make_dataset():
    from data.datasets.ca1m import CA1MDataset

    ca1m_dir = "/data1/lyq/CA1M-dataset/CA1M-dataset/training/"
    ca1m_ann = "/data1/lyq/CA1M-dataset/CA1M-dataset/"
    return CA1MDataset(
        common_conf=make_common_config(),
        split="train",
        CA1M_DIR=ca1m_dir,
        CA1M_ANNOTATION_DIR=ca1m_ann,
    )


def make_composed_dataset():
    from data.composed_dataset import ComposedDataset

    cfg = OmegaConf.create(
        {
            "dataset_configs": [
                {
                    "_target_": "data.datasets.ca1m.CA1MDataset",
                    "split": "train",
                    "CA1M_DIR": "/data1/lyq/CA1M-dataset/CA1M-dataset/training/",
                    "CA1M_ANNOTATION_DIR": "/data1/lyq/CA1M-dataset/CA1M-dataset/",
                }
            ],
        }
    )
    return ComposedDataset(dataset_configs=cfg.dataset_configs, common_config=make_common_config())


def test_step_1_2_raw_batch():
    """Step 1.2: CA1MDataset.get_data() returns qcod_scene_* numeric fields."""
    print("\n=== Step 1.2: CA1MDataset raw batch contains qcod_scene_* ===")
    ds = make_dataset()
    # Call get_data directly to skip ComposedDataset tensor conversion.
    batch = ds.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)

    required = [
        "qcod_scene_corners",
        "qcod_scene_R",
        "qcod_scene_scale",
        "qcod_scene_valid",
    ]
    for key in required:
        assert key in batch, f"missing key: {key}"

    # Explicitly NOT present (D5 forbids List[str] in batch)
    forbidden = ["qcod_scene_ids", "qcod_scene_category", "qcod_frame_ids"]
    for key in forbidden:
        assert key not in batch, f"batch must NOT contain {key} (see Plan D5)"

    # Shapes
    assert batch["qcod_scene_corners"].shape == (500, 8, 3), batch["qcod_scene_corners"].shape
    assert batch["qcod_scene_R"].shape == (500, 3, 3)
    assert batch["qcod_scene_scale"].shape == (500, 3)
    assert batch["qcod_scene_valid"].shape == (500,)
    assert batch["qcod_scene_valid"].dtype == bool

    n_valid = int(batch["qcod_scene_valid"].sum())
    print(f"  scene {batch['seq_name']}: {n_valid} valid instances")
    assert n_valid > 0, "Expected >=1 real instance after filtering"

    # Padding zone should be zeros / identity
    assert np.allclose(batch["qcod_scene_corners"][n_valid:], 0.0)
    # R padding should be identity for non-valid entries (nicer than zero for later math)
    identity = np.eye(3, dtype=np.float32)
    np.testing.assert_allclose(
        batch["qcod_scene_R"][n_valid:], np.broadcast_to(identity, (500 - n_valid, 3, 3))
    )

    print("  shapes OK, padding OK, valid count OK")
    return batch


def test_step_1_3_composed_sample():
    """Step 1.3: ComposedDataset.__getitem__ yields torch tensors for qcod fields."""
    print("\n=== Step 1.3: ComposedDataset sample is torch-ready ===")
    ds = make_composed_dataset()
    # __getitem__ expects (seq_idx, num_images, aspect_ratio) tuple
    sample = ds[(0, 4, 1.0)]

    for key in [
        "qcod_scene_corners",
        "qcod_scene_R",
        "qcod_scene_scale",
        "qcod_scene_valid",
    ]:
        assert key in sample, f"missing key in sample: {key}"
        assert isinstance(sample[key], torch.Tensor), f"{key} is not a torch.Tensor"

    assert sample["qcod_scene_corners"].dtype == torch.float32
    assert sample["qcod_scene_R"].dtype == torch.float32
    assert sample["qcod_scene_scale"].dtype == torch.float32
    assert sample["qcod_scene_valid"].dtype == torch.bool

    assert "qcod_scene_ids" not in sample, "sample must not expose string fields"

    print(
        f"  qcod_scene_corners: {tuple(sample['qcod_scene_corners'].shape)}, "
        f"valid={sample['qcod_scene_valid'].sum().item()}"
    )
    return ds, sample


def test_step_1_3_default_collate(ds):
    """default_collate can stack multiple samples (no custom collate_fn needed)."""
    print("\n=== Step 1.3: default_collate stacks a multi-sample batch ===")
    samples = [ds[(i, 4, 1.0)] for i in range(4)]
    # default_collate chokes on dicts that contain None values, so filter them out
    # (ComposedDataset sets some entries to None when tracks are disabled).
    cleaned = [{k: v for k, v in s.items() if v is not None} for s in samples]
    batch = default_collate(cleaned)

    assert batch["qcod_scene_corners"].shape == (4, 500, 8, 3)
    assert batch["qcod_scene_R"].shape == (4, 500, 3, 3)
    assert batch["qcod_scene_scale"].shape == (4, 500, 3)
    assert batch["qcod_scene_valid"].shape == (4, 500)
    assert batch["qcod_scene_valid"].dtype == torch.bool

    valid_counts = batch["qcod_scene_valid"].sum(dim=1).tolist()
    print(f"  collated batch qcod_scene valid counts per sample: {valid_counts}")
    assert all(v > 0 for v in valid_counts), "All samples should have valid instances"
    return batch


def test_step_1_4_normalize_function(batch):
    """Step 1.4: normalize_* correctly handles qcod kwargs and returns them."""
    print("\n=== Step 1.4: normalization function directly ===")
    from train_utils.normalization import normalize_camera_extrinsics_and_points_boxes_batch

    # The test batch is on CPU; the function asserts CPU device.
    out = normalize_camera_extrinsics_and_points_boxes_batch(
        extrinsics=batch["extrinsics"],
        cam_points=batch["cam_points"],
        world_points=batch["world_points"],
        depths=batch["depths"],
        point_masks=batch["point_masks"],
        bbox_corners=batch["bbox_corners"],
        scale_by_points=True,
        qcod_scene_corners=batch.get("qcod_scene_corners"),
        qcod_scene_scale=batch.get("qcod_scene_scale"),
        qcod_scene_valid=batch.get("qcod_scene_valid"),
    )
    (
        new_ext,
        new_cam,
        new_world,
        new_depths,
        new_bbox,
        new_qcod_corners,
        new_qcod_scale,
        new_qcod_valid,
    ) = out

    assert new_qcod_corners is not None
    assert new_qcod_scale is not None
    assert new_qcod_valid is not None

    # Magnitude check: after dividing by avg_scale, valid corners should be in a
    # sensible range for indoor scenes (~1-5 units typically; certainly < 30).
    valid_mask = new_qcod_valid  # [B, 500]
    valid_corners = new_qcod_corners[valid_mask]  # [N_total_valid, 8, 3]
    max_abs = valid_corners.abs().max().item()
    mean_abs = valid_corners.abs().mean().item()
    print(f"  valid corners after norm: mean |x|={mean_abs:.3f}, max |x|={max_abs:.3f}")
    assert max_abs < 50.0, (
        f"valid corners magnitude {max_abs:.2f} too large — likely coord system or "
        "scale bug upstream"
    )

    # Ratio preservation: corners/scale should be the same before and after norm.
    pre_corners = batch["qcod_scene_corners"][valid_mask]
    pre_scale = batch["qcod_scene_scale"][valid_mask]
    post_corners = new_qcod_corners[valid_mask]
    post_scale = new_qcod_scale[valid_mask]

    pre_ratio = (pre_corners.norm() / (pre_scale.norm() + 1e-8)).item()
    post_ratio = (post_corners.norm() / (post_scale.norm() + 1e-8)).item()
    print(f"  corner/scale ratio: pre={pre_ratio:.4f}, post={post_ratio:.4f}")
    # Allow ~5% slack for check_and_fix_inf_nan clamps and fp rounding.
    assert abs(pre_ratio - post_ratio) < 0.05 * max(pre_ratio, 1e-6), (
        "Corner/scale ratio changed after normalization — avg_scale mismatch?"
    )

    print("  ✓ normalization returned qcod fields, magnitudes reasonable, ratio preserved")


def test_step_1_5_trainer_writeback(batch):
    """Step 1.5: Trainer._process_batch actually unpacks and writes back qcod fields.

    Uses a minimal mock self (SimpleNamespace) to invoke _process_batch as a
    plain function without starting a real Trainer / Hydra / DDP.

    On Python classes, ``Trainer._process_batch`` is a plain function (not a
    bound method), so we pass the mock self as the first positional argument
    directly — no ``.__func__`` needed.
    """
    print("\n=== Step 1.5: Trainer._process_batch writeback ===")
    from types import SimpleNamespace

    # Importing Trainer triggers transitive imports (hydra, DDP, etc.).
    # For Phase 1 C3 this is no longer optional: if Trainer cannot be imported,
    # Step 1.5 has NOT been validated and the smoke test must fail.
    try:
        from trainer import Trainer
    except Exception as exc:
        raise AssertionError(
            "Step 1.5 requires importing Trainer and executing "
            "Trainer._process_batch(...). Import failure is a hard test failure, "
            "not a skip."
        ) from exc

    # _process_batch only reads self.data_conf.train.common_config.repeat_batch
    # (see training/trainer.py:996). Everything else is untouched. Mock just that.
    fake_self = SimpleNamespace(
        data_conf=SimpleNamespace(
            train=SimpleNamespace(
                common_config=SimpleNamespace(repeat_batch=False),
            ),
        ),
    )

    # Record pre-call identity so we can verify the tensor got replaced.
    pre_corners_id = id(batch["qcod_scene_corners"])

    # Call as plain function with mock self.
    out_batch = Trainer._process_batch(fake_self, batch)

    # Verify qcod fields are present and updated.
    assert "qcod_scene_corners" in out_batch, "missing qcod_scene_corners"
    assert "qcod_scene_scale" in out_batch, "missing qcod_scene_scale"
    assert "qcod_scene_valid" in out_batch, "missing qcod_scene_valid"

    # Tensor identity: must be a NEW tensor (normalize creates new tensors).
    assert id(out_batch["qcod_scene_corners"]) != pre_corners_id, (
        "qcod_scene_corners tensor was not replaced — _process_batch path not wired?"
    )

    # Magnitude sanity.
    valid = out_batch["qcod_scene_valid"]
    valid_corners = out_batch["qcod_scene_corners"][valid]
    if valid_corners.numel() > 0:
        max_abs = valid_corners.abs().max().item()
        mean_abs = valid_corners.abs().mean().item()
        assert max_abs < 50.0, f"corners too large after writeback: max={max_abs:.2f}"
        print(f"  Trainer._process_batch writeback OK, "
              f"valid corners mean={mean_abs:.3f}, max={max_abs:.3f}")
    else:
        print("  Trainer._process_batch writeback OK (no valid corners in this batch)")


def main():
    print("=" * 60)
    print("Phase 1 QCOD data pipeline smoke test")
    print("=" * 60)

    # Force determinism
    np.random.seed(0)
    torch.manual_seed(0)

    test_step_1_2_raw_batch()
    ds, sample = test_step_1_3_composed_sample()
    batch = test_step_1_3_default_collate(ds)
    test_step_1_4_normalize_function(batch)
    # batch was consumed by 1.4 — rebuild for 1.5 test (needs pre-normalize tensors)
    batch2 = test_step_1_3_default_collate(ds)
    test_step_1_5_trainer_writeback(batch2)

    print("\n" + "=" * 60)
    print("Phase 1 smoke test: ALL CHECKS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
