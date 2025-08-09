import numpy as np
import rerun as rr
import open3d as o3d
import cv2
def visualize_with_rerun(
    depth_maps: list,          # 深度图列表 (H x W)
    color_images: list,        # RGB图像列表 (H x W x 3)
    intrinsic: np.ndarray,     # 相机内参 (3x3)
    poses: list,               # 世界→相机的位姿 (4x4 矩阵)
    depth_scale: float = 1.0   # 深度值缩放因子
):
    # 初始化Rerun
    rr.init("pointcloud_and_cameras", spawn=True)
    # 设置世界坐标系（Y轴向上）
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)  # 设定世界坐标系[7](@ref)
    points_all = []  # 用于存储所有点云数据
    colors_all = []  # 用于存储所有颜色数据
    for frame_idx, (depth, color, T_wc) in enumerate(zip(depth_maps, color_images, poses)):

        # --- 点云生成与记录 ---
        points, colors = _depth_to_pointcloud(depth, color, intrinsic, T_wc, depth_scale)
        points_all.append(points)
        colors_all.append(colors)

        # --- 相机位姿可视化 ---
        # T_cw = np.linalg.inv(T_wc)  # 转为相机→世界变换[1](@ref)
        translation = T_wc[:3, 3]
        rotation = _matrix_to_quaternion(T_wc[:3, :3])  # 旋转矩阵→四元数
        print("translation:", translation)
        print("rotation:", rotation)
        # 记录相机位姿
        rr.log(
            f"world/cameras_{frame_idx}",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=rotation)
            )
        )
        # 添加相机视锥体[7](@ref)

        camera = rr.Pinhole(
                resolution=[depth.shape[1], depth.shape[0]],
                image_from_camera=intrinsic,
            )
  
        rr.log(f"world/cameras_{frame_idx}", camera)
        rr.log(f"world/cameras_{frame_idx}/image", rr.Image(color, opacity=0.5))
    points_all = np.concatenate(points_all, axis=0)
    colors_all = np.concatenate(colors_all, axis=0)
    rr.log(
            "world/points", 
            rr.Points3D(points_all, colors=colors_all, radii=0.005)  # 控制点云半径
        )

# 辅助函数：深度图→点云
def _depth_to_pointcloud(depth, color, K, T_wc, depth_scale):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    height, width = depth.shape
    
    # 生成网格坐标
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth.flatten() * depth_scale
    x = (u.flatten() - cx) * z / fx
    y = (v.flatten() - cy) * z / fy
    
    # 转换到世界坐标系
    points_cam = np.vstack([x, y, z]).T
    points_world = _apply_transform(points_cam, T_wc)
    
    # 过滤无效点并提取颜色
    valid = z > 0
    return points_world[valid], color.reshape(-1, 3)[valid] / 255.0

def _apply_transform(points, T):
    homo_points = np.hstack([points, np.ones((len(points), 1))])
    return (T @ homo_points.T).T[:, :3]

def _matrix_to_quaternion(R):
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    return np.array([
        (R[2,1]-R[1,2])/(4*qw),
        (R[0,2]-R[2,0])/(4*qw),
        (R[1,0]-R[0,1])/(4*qw),
        qw
    ])
    
if __name__ == "__main__":
    # 准备数据（示例）
    depth_maps = [cv2.imread("/media/lyq/temp/dataset/train-CA-1M-slam/47332320/depth/0.png",-1)/1000.0, cv2.imread("/media/lyq/temp/dataset/train-CA-1M-slam/47332320/depth/100.png",-1)/1000.0]  # 深度图
    color_images = [cv2.imread("/media/lyq/temp/dataset/train-CA-1M-slam/47332320/rgb/0.png", cv2.COLOR_BGR2RGB), cv2.imread("/media/lyq/temp/dataset/train-CA-1M-slam/47332320/rgb/100.png"), cv2.COLOR_BGR2RGB]  
    H, W=depth_maps[0].shape[0], depth_maps[0].shape[1]
    if color_images[0].shape[0]!=H:
        color_images[0]=cv2.resize(color_images[0], (W,H))
        color_images[1]=cv2.resize(color_images[1], (W,H))
    intrinsic = np.loadtxt("/media/lyq/temp/dataset/train-CA-1M-slam/47332320/K_depth.txt")  # 内参
    all_poses = np.load('/media/lyq/temp/dataset/train-CA-1M-slam/47332320/all_poses.npy')
    poses = [all_poses[0], all_poses[100]]  # 世界→相机变换矩阵

    # 启动可视化
    visualize_with_rerun(
        depth_maps, 
        color_images,
        intrinsic,
        poses,
        depth_scale=1.0  # Kinect等设备需缩放
    )