# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from train_utils.general import check_and_fix_inf_nan
from math import ceil, floor
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import l1_loss



@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(self, camera=None, depth=None, point=None, track=None, box=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.track = track
        self.box = box

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        # print("batch,",batch.keys())
        # ['seq_name', 'ids', 'images', 'depths', 'extrinsics', 'intrinsics', 'cam_points', 'world_points', 'bbox_corners', 'point_masks']
        # Camera pose loss - if pose encodings are predicted
        if "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)   
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]   
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)
        
        # Depth estimation loss - if depth maps are predicted
        if "depth" in predictions:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)

        # 3D point reconstruction loss - if world points are predicted
        if "world_points" in predictions:
            point_loss_dict = compute_point_loss(predictions, batch, **self.point)
            point_loss = point_loss_dict["loss_conf_point"] + point_loss_dict["loss_reg_point"] + point_loss_dict["loss_grad_point"]
            point_loss = point_loss * self.point["weight"]
            total_loss = total_loss + point_loss
            loss_dict.update(point_loss_dict)
        
        # if 'box_result' in predictions:
        if 'pred_corners' in predictions:
              
            pred_corners = predictions['pred_corners']
            pred_logits = predictions['pred_logits']
            # box_loss_dict = compute_box_loss(box_predictions, batch)#, **self.box)   
            box_loss_dict = compute_box_logit_loss(pred_corners, pred_logits,  batch)#, **self.box)   
            box_loss = box_loss_dict["loss_box"] * self.box["weight"]   
            total_loss = total_loss + box_loss
            loss_dict.update(box_loss_dict)

        # Tracking loss - not cleaned yet, dirty code is at the bottom of this file
        if "track" in predictions:
            raise NotImplementedError("Track loss is not cleaned up yet")
        
        loss_dict["objective"] = total_loss

        return loss_dict


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    # print(f"pred_T:{pred_pose_enc[0,:, :3]}")
    # print(f"gt_T  :{gt_pose_enc[0,:, :3]}, loss_T:{loss_T.mean()}")
    
    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']
    pred_points_conf = predictions['world_points_conf']
    gt_points = batch['world_points']
    gt_points_mask = batch['point_masks']
    
    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")
    
    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)
    
    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }
    
    return loss_dict


def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth']
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch['depths']
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)
    gt_depth_mask = batch['point_masks'].clone()   # 3D points derived from depth map, so we use the same mask

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,    
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


# ---------- Chamfer Loss ----------
def chamfer_loss(corners_pred, corners_gt):
    """
    corners_pred: [N_pos, 8, 3]
    corners_gt  : [N_pos, 8, 3]
    return scalar
    """
    if corners_pred.shape[0] == 0:
        return torch.tensor(0.0, device=corners_pred.device, requires_grad=True)

    dist = torch.cdist(corners_pred, corners_gt)          # [N_pos, 8, 8]
    chamfer = dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()
    return chamfer

# ---------- Hungarian 匹配 ----------
def hungarian_2d_matching(cost_matrix):
    """
    cost_matrix: [N_pred, N_gt]
    return pred_indices, gt_indices  (长度 = min(N_pred, N_gt))
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu())
    return torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)


# zhi rui version
@torch.no_grad()
def hungarian_match_batch(pred_line, pred_bezier, pred_circle, pred_logits, gt_curves_pts, gt_curves_ends, gt_curves_types,gt_centers,gt_radiis, **cost_kwargs):
    """
    pred_ctrls: (batch_size, P, K, D)
    gt_ctrls_list: list of B tensors, each (G_i, K, D)
    returns: list of (pred_inds, gt_inds) each as np arrays
    """
    device = pred_logits.device
    bs, num_queries = pred_line.shape[:2]
    pred_logits = pred_logits.flatten(0, 1).softmax(-1) # [batch_size * num_queries, num_classes+1]
    pred_line = pred_line.flatten(0, 1)  # # [batch_size * num_queries, 6]
    pred_circle = pred_circle.flatten(0, 1)  # # [batch_size * num_queries, 7]
    pred_bezier = pred_bezier.flatten(0, 1)  # [batch_size * num_queries, 12]

    tgt_curves_pts = torch.cat([gt_curve_pts[gt_curve_types>=0] for gt_curve_pts, gt_curve_types in zip(gt_curves_pts, gt_curves_types)])
    tgt_curves_ends = torch.cat([gt_curve_ends[gt_curve_types>=0] for gt_curve_ends, gt_curve_types in zip(gt_curves_ends, gt_curves_types)])
    gt_center = torch.cat([gt_center[gt_curve_types>=0] for gt_center, gt_curve_types in zip(gt_centers, gt_curves_types)])
    gt_radii = torch.cat([gt_radii[gt_curve_types>=0] for gt_radii, gt_curve_types in zip(gt_radiis, gt_curves_types)])

    
    tgt_ids = torch.cat([gt_curve_types[gt_curve_types>=0] for gt_curve_types in gt_curves_types])
    # Compute the classification cost. Contrary to the loss, we don't use the NLL,
    # but approximate it in 1 - proba[target class].
    # The 1 is a constant that doesn't change the matching, it can be ommitted.
    cost_class = -pred_logits[:, tgt_ids]  # [batch_size * num_queries, G]

    # sample line 
    line_samp = sample_line(pred_line, cost_kwargs["num_samples"])

    # sample circle 
    circle_samp = sample_circle(pred_circle, cost_kwargs["num_samples"]) # (B*S, T, 3)

    # sample bezier 
    bezier_samp = sample_bezier(pred_bezier, cost_kwargs["num_samples"])  # (B1, T, 3)


    LINE, BEZ, CIR = 0, 1, 2
    mask_line = (tgt_ids == LINE)
    mask_bez  = (tgt_ids == BEZ)
    mask_cir  = (tgt_ids == CIR)
    G = tgt_ids.shape[0]
    geom_cost = torch.zeros((bs*num_queries, G), device=device)
    end_cost  = torch.zeros((bs*num_queries, G), device=device)

    geom_cost = dist_loss_pair_l1(bezier_samp, tgt_curves_pts)
    end_cost = dist_loss_pair_l1(torch.stack([pred_bezier[:,0:3], pred_bezier[:,9:12]], dim=1), tgt_curves_ends)


    # Final cost matrix
    C = cost_class*cost_kwargs["w_class"] + geom_cost*cost_kwargs["w_chamfer"] + end_cost*cost_kwargs["w_chamfer"]
    C = C.view(bs, num_queries, -1).cpu()
    sizes = [len(gt_curve_types[gt_curve_types>=0]) for gt_curve_types in gt_curves_types]
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

# ---------- 1) 构造 3D Chamfer 代价矩阵 ----------
def build_3d_cost(pred_corners, gt_corners):
    """
    pred_corners: [N_pred, 8, 3]
    gt_corners  : [N_gt, 8, 3]
    return      : [N_pred, N_gt]  每个元素是 (pred_i, gt_j) 的 Chamfer 距离
    """
    N_pred, N_gt = pred_corners.shape[0], gt_corners.shape[0]
    cost = torch.zeros(N_pred, N_gt, device=pred_corners.device)
    # print("pred_corners", pred_corners.shape)
    # print("gt_corners", gt_corners.shape)
    for i in range(N_pred):
        for j in range(N_gt):
            # 两个 8×3 点集之间的 Chamfer
            dist = torch.cdist(pred_corners[i], gt_corners[j])   # [8,8]
            chamfer_ij = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
            cost[i, j] = chamfer_ij
    return cost

# # ---------- 1) 构造 3D Chamfer 和 logits 的代价矩阵 ----------
# def build_3d_cost_logits(pred_corners, gt_corners, pred_logits, cost_kwargs):
#     """
#     pred_corners: [N_pred, 8, 3]
#     gt_corners  : [N_gt, 8, 3] 
#     pred_logits:  [N_pred, 2]  每个预测框的 logits, 0维度是背景概率, 1维度是前景概率
#     logit_weight: weight for logit cost component
#     return      : [N_pred, N_gt]  每个元素是 (pred_i, gt_j) 的 Chamfer 距离 + logit cost
#     """
#     N_pred, N_gt = pred_corners.shape[0], gt_corners.shape[0]
#     cost = torch.zeros(N_pred, N_gt, device=pred_corners.device)
    
#     for i in range(N_pred):
#         for j in range(N_gt):
#             # 两个 8×3 点集之间的 Chamfer
#             dist = torch.cdist(pred_corners[i], gt_corners[j])   # [8,8]
#             chamfer_ij = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
            
#             # 添加 logit cost - 对于 GT box，我们希望前景概率高（logits[1]大）
#             # 所以 logit cost = -pred_logits[i, 1] （负号表示前景概率越高，cost 越小）
#             logit_cost = -pred_logits[i, 1]  # 前景logit越大，cost越小
#             1
#             cost[i, j] = cost_kwargs['chamfer_weight'] * chamfer_ij + cost_kwargs['class_weight'] * logit_cost
#     return cost

# ---------- 1) 构造 3D Chamfer 和 logits 的代价矩阵 (vectorized) ----------
def build_3d_cost_logits(pred_corners, gt_corners, pred_logits, cost_kwargs):
    """
    Vectorized cost matrix computation (no Python for-loops).

    pred_corners: [N_pred, 8, 3]
    gt_corners  : [N_gt, 8, 3]
    pred_logits : [N_pred, 2] (background, foreground)
    cost = chamfer_weight * chamfer_distance + class_weight * (-foreground_logit)
    return: [N_pred, N_gt]
    """
    device = pred_corners.device
    N_pred, N_gt = pred_corners.shape[0], gt_corners.shape[0]

    if N_pred == 0 or N_gt == 0:
        return torch.zeros(N_pred, N_gt, device=device)

    # Pairwise point distances for every (pred_box, gt_box)
    # diff: [N_pred, N_gt, 8, 8, 3]
    diff = pred_corners[:, None, :, None, :] - gt_corners[None, :, None, :, :]
    dist = torch.norm(diff, dim=-1)  # [N_pred, N_gt, 8, 8]

    # Chamfer components
    # For each pred point find nearest gt point
    min_pred = dist.min(dim=3)[0].mean(dim=2)  # [N_pred, N_gt]
    # For each gt point find nearest pred point
    min_gt = dist.min(dim=2)[0].mean(dim=2)    # [N_pred, N_gt]
    chamfer = min_pred + min_gt                # [N_pred, N_gt]

    # Classification (foreground encourages lower cost)
    logit_cost = -(pred_logits[:, 1].sigmoid()).unsqueeze(1).expand(N_pred, N_gt)

    cost = cost_kwargs['chamfer_weight'] * chamfer + cost_kwargs['class_weight'] * logit_cost
    return cost

# ---------- 总接口 ----------
def compute_box_loss_single(
        pred_corners,   # [N_pred, 8, 3]
        gt_corners,     # [N_gt, 8, 3]
        pred_2d=None,        # [N_pred, 4]  用于匹配
        gt_2d=None,          # [N_gt, 4]
        cost_bbox_weight=5.0):
    """
    1. 用 2D box 做匈牙利匹配
    2. 匹配成功的角点用 Chamfer 损失
    """
    N_pred, N_gt = pred_corners.shape[0], gt_corners.shape[0]
    if N_pred == 0 or N_gt == 0:
        return torch.tensor(0.0, device=pred_corners.device, requires_grad=True)

    # 1) 构造 2D L1 代价矩阵
    # cost_bbox = torch.cdist(pred_2d, gt_2d, p=1)  # [N_pred, N_gt]
    # print("pred_corners", pred_corners.shape)
    # print("gt_corners", gt_corners.shape)
    cost_bbox = build_3d_cost(pred_corners, gt_corners)
    
    # 2) 匈牙利匹配
    pred_idx, gt_idx = hungarian_2d_matching(cost_bbox)

    # 3) 取出匹配上的角点
    matched_pred = pred_corners[pred_idx]
    matched_gt   = gt_corners[gt_idx]

    # 4) Chamfer Loss
    loss = chamfer_loss(matched_pred, matched_gt)
    
    
    # if loss > 100:
    #     print("matched_pred", matched_pred[0],matched_pred.shape)
    #     print("matched_gt", matched_gt[0],matched_gt.shape)
    #     print("\n")
    
    return loss


# ---------- 总接口 ----------
def compute_box_logit_loss_single(
    pred_corners,   # [N_pred, 8, 3]
    pred_logits,    # [N_pred, 2]
    gt_corners,     # [N_gt, 8, 3]
    pred_2d=None,        # [N_pred, 4]  用于匹配
    gt_2d=None,          # [N_gt, 4]
    w_box = 1.0, # Chamfer loss weight
    w_class = 1.0, # Classification loss weight
    ):
    """
    1. 用 3D Chamfer 和 logits 做匈牙利匹配
    2. 匹配成功的角点用 Chamfer 损失
    3. 对匹配上的box使用前景分类loss，未匹配上的使用背景分类loss
    """
    N_pred, N_gt = pred_corners.shape[0], gt_corners.shape[0]
    
    # 创建分类标签：所有预测框初始为背景(0)
    class_labels = torch.zeros(N_pred, device=pred_corners.device)
    
    chamfer_loss_val = torch.tensor(0.0, device=pred_corners.device, requires_grad=True)
    

    # 1) 构造代价矩阵（包含 Chamfer 距离和 logit 代价）
    cost_kwargs = dict(chamfer_weight=1.0, class_weight=1.0) #10.0 1.0
    cost_bbox = build_3d_cost_logits(pred_corners, gt_corners, pred_logits, cost_kwargs)
    
    # 2) 匈牙利匹配
    pred_idx, gt_idx = hungarian_2d_matching(cost_bbox)

    # 3) 取出匹配上的角点并计算 Chamfer Loss
    if len(pred_idx) > 0:
        matched_pred = pred_corners[pred_idx]
        matched_gt = gt_corners[gt_idx]
        chamfer_loss_val = chamfer_loss(matched_pred, matched_gt)
        
        # 4) 将匹配上的预测框标记为前景(0)
        class_labels[pred_idx] = 1.0

    # 5) 分类损失：对所有预测框计算交叉熵损失
    # class_loss = F.cross_entropy(pred_logits, class_labels)
    
    # 5) 二分类：取前景单一logit，做 sigmoid 再算 BCE
    # 使用 logit 差值增强数值稳定性 (等价于对前景概率做 sigmoid)
    
    # pred_prob = torch.sigmoid(pred_logits) # [N_pred]
    # class_loss = F.binary_cross_entropy(pred_prob, class_labels)
    
    # num_classes = 2
    # class_labels_one_hot = F.one_hot(class_labels, num_classes=num_classes).float()  # 转换为float类型
    # 现在 class_labels_one_hot 的形状是 [100, 2]
    # 然后计算损失
    pos_logits = pred_logits[:, 1]  # [100]
    num_pos = class_labels.sum().clamp(min=1)
    num_neg = (class_labels.numel() - num_pos)
    pos_weight = num_neg / num_pos 
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    class_loss = bce(pos_logits, class_labels)
        
    
    # 6) 总损失
    loss = w_box * chamfer_loss_val + w_class * class_loss
    
    return loss, w_box * chamfer_loss_val, w_class * class_loss

def compute_box_loss(predictions, batch):
    
    total_loss = 0
    N_seq = len(predictions)
    # calcudate loss for each sequence
    for i in range(N_seq):
        cur_predictions = predictions[i]
        N_imgs = len(cur_predictions)

        gt_box_corners_seq = batch['bbox_corners'][i] # [N_gt, 8 ,3]
        gt_box_corners_seq_sum = gt_box_corners_seq.sum(dim=[1, 2]) #[500]
        gt_box_mask = gt_box_corners_seq_sum != 0.0
        gt_box_corners_seq = gt_box_corners_seq[gt_box_mask]  # [N_gt, 8, 3]

        pred_box_corners = cur_predictions

        gt_box_corners = gt_box_corners_seq #gt_box_corners 
        
        loss = compute_box_loss_single(pred_box_corners,gt_box_corners)
        # loss = chamfer_loss(pred_box_corners, gt_box_corners) 
 
        total_loss += loss

    loss_dict = {
        f"loss_box": total_loss,
    }

    return loss_dict


def compute_box_logit_loss(pred_corners, pred_logits, batch):
    
    total_loss = 0
    total_chamfer_loss = 0
    total_class_loss = 0
    N_seq = len(pred_corners)
    # calcudate loss for each sequence
    # TODO: accelerate this
    for i in range(N_seq):
        pred_box_corners = pred_corners[i]
        pred_box_logits = pred_logits[i]
        N_imgs = len(pred_box_corners)

        gt_box_corners_seq = batch['bbox_corners'][i] # [N_gt, 8 ,3]
        gt_box_corners_seq_sum = gt_box_corners_seq.sum(dim=[1, 2]) #[500]
        gt_box_mask = gt_box_corners_seq_sum != 0.0
        gt_box_corners_seq = gt_box_corners_seq[gt_box_mask]  # [N_gt, 8, 3]


        gt_box_corners = gt_box_corners_seq #gt_box_corners 
        
        loss, chamfer_loss_val, class_loss = compute_box_logit_loss_single(pred_box_corners, pred_box_logits, gt_box_corners, w_box=1.0, w_class=1.0) #w_class=0.05
        # loss = chamfer_loss(pred_box_corners, gt_box_corners) 

        total_loss += loss
        total_chamfer_loss += chamfer_loss_val
        total_class_loss += class_loss
        
    loss_dict = {
        f"loss_box": total_loss,
        f"loss_chamfer": total_chamfer_loss,
        f"loss_class": total_class_loss,
    }

    return loss_dict
    
########################################################################################
########################################################################################

# Dirty code for tracking loss:

########################################################################################
########################################################################################

'''
def _compute_losses(self, coord_preds, vis_scores, conf_scores, batch):
    """Compute tracking losses using sequence_loss"""
    gt_tracks = batch["tracks"]  # B, S, N, 2
    gt_track_vis_mask = batch["track_vis_mask"]  # B, S, N

    # if self.training and hasattr(self, "train_query_points"):
    train_query_points = coord_preds[-1].shape[2]
    gt_tracks = gt_tracks[:, :, :train_query_points]
    gt_tracks = check_and_fix_inf_nan(gt_tracks, "gt_tracks", hard_max=None)

    gt_track_vis_mask = gt_track_vis_mask[:, :, :train_query_points]

    # Create validity mask that filters out tracks not visible in first frame
    valids = torch.ones_like(gt_track_vis_mask)
    mask = gt_track_vis_mask[:, 0, :] == True
    valids = valids * mask.unsqueeze(1)



    if not valids.any():
        print("No valid tracks found in first frame")
        print("seq_name: ", batch["seq_name"])
        print("ids: ", batch["ids"])
        print("time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        dummy_coord = coord_preds[0].mean() * 0          # keeps graph & grads
        dummy_vis = vis_scores.mean() * 0
        if conf_scores is not None:
            dummy_conf = conf_scores.mean() * 0
        else:
            dummy_conf = 0
        return dummy_coord, dummy_vis, dummy_conf                # three scalar zeros


    # Compute tracking loss using sequence_loss
    track_loss = sequence_loss(
        flow_preds=coord_preds,
        flow_gt=gt_tracks,
        vis=gt_track_vis_mask,
        valids=valids,
        **self.loss_kwargs
    )

    vis_loss = F.binary_cross_entropy_with_logits(vis_scores[valids], gt_track_vis_mask[valids].float())

    vis_loss = check_and_fix_inf_nan(vis_loss, "vis_loss", hard_max=None)


    # within 3 pixels
    if conf_scores is not None:
        gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < 3
        conf_loss = F.binary_cross_entropy_with_logits(conf_scores[valids], gt_conf_mask[valids].float())
        conf_loss = check_and_fix_inf_nan(conf_loss, "conf_loss", hard_max=None)
    else:
        conf_loss = 0

    return track_loss, vis_loss, conf_loss



def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom.clamp(min=1)
    mean = torch.where(denom > 0,
                       mean,
                       torch.zeros_like(mean))
    return mean


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8, vis_aware=False, huber=False, delta=10, vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=None)

        i_loss = torch.mean(i_loss, dim=3) # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)

        num_valid_points = combined_mask.sum()

        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)  # Add, don't add to the mask itself.
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if num_valid_points > 2:
                i_loss = i_loss[combined_mask]
                flow_loss += i_weight * i_loss.mean()
            else:
                i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_safe_check_{i}", hard_max=None)
                flow_loss += 0 * i_loss.mean()

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss
'''


