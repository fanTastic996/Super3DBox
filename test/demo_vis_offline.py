import numpy as np
import rerun as rr
import open3d as o3d
import cv2
import os
import argparse
from scipy.spatial.transform import Rotation
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
import open3d as o3d
import json
from scipy.spatial import KDTree
from utils import *
import torch
import pickle


def visualize_with_rerun(
    depth_maps: list,          # 深度图列表 (H x W)
    color_images: list,        # RGB图像列表 (H x W x 3)
    intrinsics: np.ndarray,     # 相机内参 (3x3)
    poses: np.ndarray,               # 世界→相机的位姿 (4x4 矩阵)
    depth_scale: float = 1.0,   # 深度值缩放因子
    json_path: str = None,  # 可选的 JSON 文件路径，用于加载边界框数据
    scene_path: str = None,  # 场景数据路径，用于加载GT点云和相机参数
    image_idxs: list = None,  # 可选的图像索引列表，用于过滤可视化的边界框
    predictions: dict = None  # 可选的预测结果字典，用于可视化预测边界框
):
    # 初始化Rerun
    rr.init("pointcloud_and_cameras", spawn=True)
    # 设置世界坐标系（Y轴向上）
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)  # 设定世界坐标系[7](@ref)
    points_all = []  # 用于存储所有点云数据
    colors_all = []  # 用于存储所有颜色数据
    for frame_idx, (depth, color, T_wc) in enumerate(zip(depth_maps, color_images, poses)):
        frameid = image_idxs[frame_idx]
        intrinsic = intrinsics[frame_idx]
        print('depth.shape', depth.shape)
        print('color.shape', color.shape)
        print('T_wc', T_wc.shape)
        print('intrinsic', intrinsic.shape)
        # --- 点云生成与记录 ---
        points, colors = depth_to_pointcloud(depth, color, intrinsic, T_wc, depth_scale)
        points_all.append(points)
        colors_all.append(colors)

        # --- 相机位姿可视化 ---
        # T_cw = np.linalg.inv(T_wc)  # 转为相机→世界变换[1](@ref)
        translation = T_wc[:3, 3]
        rotation = matrix_to_quaternion(T_wc[:3, :3])  # 旋转矩阵→四元数
        # 添加相机视锥体[7](@ref)
        camera = rr.Pinhole(
                resolution=[depth.shape[1], depth.shape[0]],
                image_from_camera=intrinsic,
            )
        rr.log(f"world/cameras_{frameid}", camera)
        # 记录相机位姿
        rr.log(
            f"world/cameras_{frameid}",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=rotation)
            )
        )
        # 记录相机图像
        rr.log(f"world/cameras_{frameid}/image", rr.Image(color, opacity=0.5))

    points_all = np.concatenate(points_all, axis=0)
    colors_all = np.concatenate(colors_all, axis=0)
    rr.log(
            "world/points", 
            rr.Points3D(points_all, colors=colors_all, radii=0.005)  # 控制点云半径
        )
    
    #visualize the bbox
    visualize_bboxes_from_pred(predictions)


    
if __name__ == "__main__":
    
    #python demo_vis_offline.py --scene_id 43649409 --frame_ids 478,546  --pred /home/lyq/myprojects/vggt_new/vggt/vis_results/43649409_pred.pkl
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="点云与相机位姿可视化")
    parser.add_argument("--scene_id", type=str, required=True, 
                        help="场景ID (例如:47332320)")
    parser.add_argument("--frame_ids", type=str, required=True, 
                        help="图像ID列表，逗号分隔 (例如:0,100,200)")
    parser.add_argument("--data_root", type=str, default="/media/lyq/temp/dataset/train-CA-1M-slam", 
                        help="数据集根目录")
    parser.add_argument("--pred", type=str, default=None, 
                        help="预测结果路径")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    if args.pred is not None:
        # 如果提供了预测结果路径，则加载预测结果
        with open(args.pred, 'rb') as f:
            loaded_data = pickle.load(f)  # 反序列化
            predictions = loaded_data  # 使用加载的数据作为预测结果
    else:
        predictions = None
    
    # 准备路径
    scene_path = os.path.join(args.data_root, args.scene_id)
    depth_dir = os.path.join(scene_path, "depth")
    rgb_dir = os.path.join(scene_path, "rgb")


    
    # 检查路径是否存在
    for path in [depth_dir, rgb_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"路径不存在: {path}")
    
    # 加载相机内参
    intrinsic = predictions['intrinsic'][0]  # np.loadtxt(intrinsic_path)
    
    # 加载所有位姿
    all_poses = predictions['extrinsic'][0] #np.load(poses_path)
    # 创建扩展行 [0,0,0,1]，形状 [2, 1, 4]
    extension = np.zeros((all_poses.shape[0], 1, all_poses.shape[2]))
    extension[:, :, -1] = 1  # 最后一列赋值为1

    # 沿行方向拼接（axis=1）
    all_poses = np.concatenate((all_poses, extension), axis=1) # [N,4,4]
    print
    
    # 处理帧ID列表
    frame_ids = [int(id_str.strip()) for id_str in args.frame_ids.split(",")]
    
    # 准备数据
    depth_maps = []
    color_images = []
    
    # 加载每个帧的数据
    for frame_id in frame_ids:
        # 深度图路径
        depth_path = os.path.join(depth_dir, f"{frame_id}.png")
        if not os.path.exists(depth_path):
            print(f"警告: 深度图不存在 {depth_path}, 跳过")
            continue
        
        # RGB图像路径
        rgb_path = os.path.join(rgb_dir, f"{frame_id}.png")
        if not os.path.exists(rgb_path):
            print(f"警告: RGB图像不存在 {rgb_path}, 跳过")
            continue
        
        # 加载深度图（除以1000转换为米）
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"警告: 无法加载深度图 {depth_path}, 跳过")
            continue
        depth = depth.astype(np.float32) / 1000.0
        depth_maps.append(depth)
        
        # 加载RGB图像并转换为RGB格式
        color = cv2.imread(rgb_path)
        if color is None:
            print(f"警告: 无法加载RGB图像 {rgb_path}, 跳过")
            continue
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)  # 转换为RGB
        
        # 调整RGB图像尺寸以匹配深度图
        H, W = depth.shape[:2]
        if color.shape[0] != H or color.shape[1] != W:
            color = cv2.resize(color, (W, H))
        color_images.append(color)
        
        
    
    # 检查是否有有效数据
    if not depth_maps or not color_images :
        raise ValueError("没有有效的图像，请检查输入参数")
    
    # 启动可视化
    visualize_with_rerun(
        depth_maps, 
        color_images,
        intrinsic,
        all_poses,
        depth_scale=1.0,
        json_path=os.path.join(scene_path, "instances.json"),
        scene_path=scene_path,
        image_idxs=frame_ids,
        predictions=predictions
    )
    
    