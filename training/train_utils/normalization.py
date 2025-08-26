# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
from typing import Optional, Tuple
from vggt.utils.geometry import closed_form_inverse_se3
from train_utils.general import check_and_fix_inf_nan


def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")


def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = True,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    # 验证输入张量是否有效（例如检查NaN/Inf）
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")

    # 获取批次大小(B)和序列长度(S)，设备设为CPU（断言确认）
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    assert device == torch.device("cpu")


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    # 将3x4外参矩阵转换为4x4齐次形式（在倒数第二维添加零行）
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    # 设置齐次坐标的缩放因子（最后一行设置为[0,0,0,1]）
    extrinsics_homog[:, :, -1, -1] = 1.0

    # 计算首帧相机外参的逆矩阵（将坐标系变换到首帧相机坐标系）
    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    # 将所有相机外参左乘首帧相机的逆矩阵，实现坐标系统一
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)

    # 处理世界坐标系下的3D点
    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        # 提取首帧相机的旋转矩阵(R)和平移向量(t)
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None

    # 缩放控制：根据点云距离归一化场景尺度
    if scale_by_points:
        # 创建相机坐标点和深度的拷贝（避免修改原始数据）
        new_cam_points = cam_points.clone()
        new_depths = depths.clone()
        # 计算有效点在首帧相机坐标系下的欧氏距离
        dist = new_world_points.norm(dim=-1)
        # 根据点掩码计算有效距离总和及有效点数量
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        # 计算平均距离（分母添加小量防除零）
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)

        # 用平均距离缩放世界坐标点
        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        # 缩放相机外参的平移分量
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        # 缩放深度图和相机坐标点（如果存在）
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        # 不缩放时直接返回原尺寸数据（3x4外参格式）
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths
    # 将4x4外参矩阵截断回3x4格式
    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    # 检查并修复输出张量的异常值（NaN/Inf）
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)

    # 返回归一化后的四元组结果
    return new_extrinsics, new_cam_points, new_world_points, new_depths





# def normalize_camera_extrinsics_and_points_boxes_batch(
#     extrinsics: torch.Tensor,
#     cam_points: Optional[torch.Tensor] = None,
#     world_points: Optional[torch.Tensor] = None,
#     depths: Optional[torch.Tensor] = None,
#     scale_by_points: bool = True,
#     point_masks: Optional[torch.Tensor] = None,
#     bbox_corners: Optional[torch.Tensor] = None,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
#     """
#     Normalize camera extrinsics and corresponding 3D points.
    
#     This function transforms the coordinate system to be centered at the first camera
#     and optionally scales the scene to have unit average distance.
    
#     Args:
#         extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
#         cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
#         world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
#         depths: Depth maps of shape (B, S, H, W)
#         scale_by_points: Whether to normalize the scale based on point distances
#         point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
#     Returns:
#         Tuple containing:
#         - Normalized camera extrinsics of shape (B, S, 3, 4)
#         - Normalized camera points (same shape as input cam_points)
#         - Normalized world points (same shape as input world_points)
#         - Normalized depths (same shape as input depths)
#     """
#     # Validate inputs
#     # 验证输入张量是否有效（例如检查NaN/Inf）
#     check_valid_tensor(extrinsics, "extrinsics")
#     check_valid_tensor(cam_points, "cam_points")
#     check_valid_tensor(world_points, "world_points")
#     check_valid_tensor(depths, "depths")
#     check_valid_tensor(bbox_corners, "bbox_corners")

#     # 获取批次大小(B)和序列长度(S)，设备设为CPU（断言确认）
#     B, S, _, _ = extrinsics.shape
#     device = extrinsics.device
#     assert device == torch.device("cpu")


#     # Convert extrinsics to homogeneous form: (B, N,4,4)
#     # 将3x4外参矩阵转换为4x4齐次形式（在倒数第二维添加零行）
#     extrinsics_homog = torch.cat(
#         [
#             extrinsics,
#             torch.zeros((B, S, 1, 4), device=device),
#         ],
#         dim=-2,
#     )
#     # 设置齐次坐标的缩放因子（最后一行设置为[0,0,0,1]）
#     extrinsics_homog[:, :, -1, -1] = 1.0

#     # 计算首帧相机外参的逆矩阵（将坐标系变换到首帧相机坐标系）
#     # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
#     # which can be also viewed as the cam_to_world extrinsic matrix
#     first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
#     # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
#     # 将所有相机外参左乘首帧相机的逆矩阵，实现坐标系统一
#     new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)

#     # 处理世界坐标系下的3D点
#     if world_points is not None:
#         # since we are transforming the world points to the first camera's coordinate system
#         # we directly use the cam_from_world extrinsic matrix of the first camera
#         # instead of using the inverse of the first camera's extrinsic matrix
#         # 提取首帧相机的旋转矩阵(R)和平移向量(t)
#         R = extrinsics[:, 0, :3, :3]
#         t = extrinsics[:, 0, :3, 3]
#         new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        
#         bbox_corners_sum = bbox_corners.sum(dim=[-2, -1]) 
#         padding_bbox_mask = bbox_corners_sum == 0.0 # [B, N_box]
#         # (B, N_box, 8, 3) @ (B, 1, 3, 3) -> (B, N_box, 8, 3)
#         new_bbox_corners = (bbox_corners @ R.transpose(-1, -2).unsqueeze(1)) + t.unsqueeze(1).unsqueeze(1)



#     # 缩放控制：根据点云距离归一化场景尺度
#     if scale_by_points:
#         # 创建相机坐标点和深度的拷贝（避免修改原始数据）
#         new_cam_points = cam_points.clone()
#         new_depths = depths.clone()
#         # 计算有效点在首帧相机坐标系下的欧氏距离
#         dist = new_world_points.norm(dim=-1)
#         # 根据点掩码计算有效距离总和及有效点数量
#         dist_sum = (dist * point_masks).sum(dim=[1,2,3])
#         valid_count = point_masks.sum(dim=[1,2,3])
#         # 计算平均距离（分母添加小量防除零）
#         avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)
        
#         # 用平均距离缩放世界坐标点
#         new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
#         # 用平均距离缩放世界bbox角点
#         new_bbox_corners = new_bbox_corners / avg_scale.view(-1, 1, 1, 1)
        
        
#         # 缩放相机外参的平移分量
#         new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
#         # 缩放深度图和相机坐标点（如果存在）
#         if depths is not None:
#             new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
#         if cam_points is not None:
#             new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
#     else:
#         # 不缩放时直接返回原尺寸数据（3x4外参格式）
#         return new_extrinsics[:, :, :3], cam_points, new_world_points, depths, new_bbox_corners
    
#     #mask those boxes that are padding due to batch stack
#     new_bbox_corners[padding_bbox_mask] = 0.0
    
#     # 将4x4外参矩阵截断回3x4格式
#     new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
#     # 检查并修复输出张量的异常值（NaN/Inf）
#     new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
#     new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
#     new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
#     new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)
#     new_bbox_corners = check_and_fix_inf_nan(new_bbox_corners, "new_bbox_corners", hard_max=None)
    
#     new_bbox_corners[new_bbox_corners>100]=0.0
#     new_bbox_corners[new_bbox_corners<-100]=0.0

#     # 返回归一化后的四元组结果
#     return new_extrinsics, new_cam_points, new_world_points, new_depths, new_bbox_corners



def normalize_camera_extrinsics_and_points_boxes_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = False,
    point_masks: Optional[torch.Tensor] = None,
    bbox_corners: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    # 验证输入张量是否有效（例如检查NaN/Inf）
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")
    check_valid_tensor(bbox_corners, "bbox_corners")

    # 获取批次大小(B)和序列长度(S)，设备设为CPU（断言确认）
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    assert device == torch.device("cpu")


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    # 将3x4外参矩阵转换为4x4齐次形式（在倒数第二维添加零行）
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    # 设置齐次坐标的缩放因子（最后一行设置为[0,0,0,1]）
    extrinsics_homog[:, :, -1, -1] = 1.0

    # 计算首帧相机外参的逆矩阵（将坐标系变换到首帧相机坐标系）
    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    # 将所有相机外参左乘首帧相机的逆矩阵，实现坐标系统一
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)
    new_extrinsics = torch.matmul(first_cam_extrinsic_inv.unsqueeze(1),extrinsics_homog)  # (B,N,4,4)
    
    
    # 处理世界坐标系下的3D点
    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        # 提取首帧相机的旋转矩阵(R)和平移向量(t)
        R = first_cam_extrinsic_inv[:, :3, :3] #extrinsics[:, 0, :3, :3]
        t = first_cam_extrinsic_inv[:, :3, 3] #extrinsics[:, 0, :3, 3]
        new_world_points = (R.unsqueeze(1).unsqueeze(2) @ (world_points.permute(0,1,2,4,3))).permute(0,1,2,4,3) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        
        bbox_corners_sum = bbox_corners.sum(dim=[-2, -1]) 
        padding_bbox_mask = bbox_corners_sum == 0.0 # [B, N_box]
        # (B, N_box, 8, 3) @ (B, 1, 3, 3) -> (B, N_box, 8, 3)
        # new_bbox_corners = (bbox_corners @ R.transpose(-1, -2).unsqueeze(1)) + t.unsqueeze(1).unsqueeze(1)
        new_bbox_corners = (R.unsqueeze(1) @ (bbox_corners.permute(0,1,3,2))).permute(0,1,3,2) + t.unsqueeze(1).unsqueeze(1)


    new_cam_points = cam_points.clone()
    new_depths = depths.clone()
    # 缩放控制：根据点云距离归一化场景尺度
    if scale_by_points:
        # 创建相机坐标点和深度的拷贝（避免修改原始数据）

        # 计算有效点在首帧相机坐标系下的欧氏距离
        dist = new_world_points.norm(dim=-1)
        # 根据点掩码计算有效距离总和及有效点数量
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        # 计算平均距离（分母添加小量防除零）
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)
        
        # 用平均距离缩放世界坐标点
        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        # 用平均距离缩放世界bbox角点
        new_bbox_corners = new_bbox_corners / avg_scale.view(-1, 1, 1, 1)
        
        
        # 缩放相机外参的平移分量
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        # 缩放深度图和相机坐标点（如果存在）
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    # else:
    #     # 不缩放时直接返回原尺寸数据（3x4外参格式）
    #     return new_extrinsics[:, :, :3], cam_points, new_world_points, depths, new_bbox_corners
    
    #mask those boxes that are padding due to batch stack
    new_bbox_corners[padding_bbox_mask] = 0.0
    
    # 将4x4外参矩阵截断回3x4格式
    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    # 检查并修复输出张量的异常值（NaN/Inf）
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)
    new_bbox_corners = check_and_fix_inf_nan(new_bbox_corners, "new_bbox_corners", hard_max=None)
    
    new_bbox_corners[new_bbox_corners>100]=0.0
    new_bbox_corners[new_bbox_corners<-100]=0.0

    # 返回归一化后的四元组结果
    return new_extrinsics, new_cam_points, new_world_points, new_depths, new_bbox_corners

