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



def load_depth_maps(depth_path, image_indices, scale=0.001):
    """按需加载指定的深度图，统一尺寸"""
    depth_maps = []

    for idx in image_indices:
        depth_file = os.path.join(depth_path, f'{idx}.png')
        if not os.path.exists(depth_file):
            print(f"Warning: Depth file not found: {depth_file}, using zeros")
            # 使用零填充的深度图
            depth_maps.append(np.zeros((640, 480), dtype=np.uint16))
            continue
        try:
            d = cv2.imread(depth_file, -1) * scale
            depth_array = np.asarray(d)
            depth_maps.append(depth_array)

        except Exception as e:
            print(f"Warning: Error loading depth map {depth_file}: {str(e)}, using zeros")
            depth_maps.append(np.zeros((640, 480), dtype=np.uint16))

    return np.stack(depth_maps, axis=0)

def filter_gt_boxes_for_images(scene_data, image_indices, points_all):
    """为指定图像过滤GT boxes"""
    corners_array = scene_data['corners_array']
    K = scene_data['K']

    # try:
    poses = scene_data['poses'][image_indices]
    depth_maps = load_depth_maps(
        scene_data['depth_path'],
        image_indices,
        scale=0.001
    )

    # 使用filter_gt_boxes.py的逻辑
    filtered_corners, mask = filter_3d_corners(
        corners_array, K, poses, depth_maps, points_all,
        frustum_threshold=4, dist_threshold=0.2
    )

    return filtered_corners, mask

def frustum_culling_bbox_level(corners, K, poses, depth_maps, near, far, threshold=2):
    """视锥体剔除"""
    N = corners.shape[0]
    M = len(poses)
    bbox_mask = np.zeros((N, 8), dtype=bool)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    for i in range(M):
        try:
            pose = poses[i]
            depth_map = depth_maps[i]
            H, W = depth_map.shape

            pose_inv = np.linalg.inv(pose)

            # 变换到相机坐标系
            hom_corners = np.concatenate([corners, np.ones((N, 8, 1))], axis=-1)
            cam_points = np.dot(hom_corners, pose_inv.T)

            x, y, z = cam_points[..., 0], cam_points[..., 1], cam_points[..., 2]

            # 深度检查
            valid_z = (z > near) & (z < far)

            # 投影检查
            with np.errstate(divide='ignore', invalid='ignore'):
                u = (fx * x / z + cx).astype(int)
                v = (fy * y / z + cy).astype(int)

            valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            # print("valid_uv", valid_uv.shape, valid_uv.dtype, valid_uv.sum())
            # print("valid_z",valid_z.shape, valid_z.dtype, valid_z.sum())
            valid_corner = valid_z & valid_uv
            # print("valid_corner", valid_corner.shape, valid_corner.dtype, valid_corner.sum())
            bbox_mask |= valid_corner
            
        except Exception as e:
            print(f"Warning: Error processing frame {i}: {str(e)}")
            continue
    
    # 至少N个角点可见
    count_visible = np.sum(bbox_mask, axis=1)
    bbox_visible = count_visible >= threshold
    print("视锥保留", bbox_visible.sum(),'/', len(bbox_visible))
    return bbox_visible

def check_bbox_proximity(bboxes, gt_points, threshold=0.5):
    """点云邻近性检查"""
    if len(bboxes) == 0 or len(gt_points) == 0:
        return np.array([], dtype=bool)

    tree = KDTree(gt_points)
    proximity_mask = np.zeros(len(bboxes), dtype=bool)

    for i, bbox in enumerate(bboxes):
        try:
            dists, _ = tree.query(bbox, k=1)
            near_points_count = np.sum(dists < threshold)

            # 至少4个角点满足条件
            if near_points_count >= 4:
                proximity_mask[i] = True
        except Exception as e:
            print(f"Warning: Error in proximity check for bbox {i}: {str(e)}")
            # 如果检查失败，默认为True
            proximity_mask[i] = True
    print("邻近性检查通过", proximity_mask.sum(), '/', len(proximity_mask))
    return proximity_mask

def filter_3d_corners(corners, K, poses, depth_maps, gt_points,
                          near=0.1, far=100.0, frustum_threshold=4, dist_threshold=0.5):  # 放宽距离阈值
    """基于filter_gt_boxes.py的过滤逻辑"""
    try:
        # 第一步：视锥体剔除
        bbox_frustum_mask = frustum_culling_bbox_level(
            corners, K, poses, depth_maps, near, far, threshold=frustum_threshold
        )
        
        visible_bboxes = corners[bbox_frustum_mask]

        if len(visible_bboxes) == 0:
            # 如果没有可见的boxes，返回空结果
            return np.empty((0, 8, 3)), np.zeros(len(corners), dtype=bool)

        # 第二步：点云邻近性验证
        bbox_proximity_mask = check_bbox_proximity(
            visible_bboxes, gt_points, dist_threshold
        )
        # bbox_proximity_mask[:] = 1 # 这里假设所有可见的bbox都满足邻近性条件

        # 组合掩码
        final_mask = np.zeros(len(corners), dtype=bool)
        final_mask[bbox_frustum_mask] = bbox_proximity_mask

        filtered_corners = corners[final_mask]
        # print("filtered_corners",filtered_corners.shape)
        final_mask_frameid = [i for i in range(final_mask.shape[0]) if final_mask[i]]
        # print( "过滤后保留的GT boxes数量:", len(filtered_corners), "对应的box ID:", final_mask_frameid)
        return filtered_corners, final_mask
    
    except Exception as e:
        print(f"Warning: Error in filter_3d_corners: {str(e)}")
        # 返回所有corners作为后备
        return corners

# 辅助函数：深度图→点云
def depth_to_pointcloud(depth, color, K, T_wc, depth_scale):
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
    points_world = apply_transform(points_cam, T_wc)
    
    # 过滤无效点并提取颜色
    valid = z > 0
    return points_world[valid], color.reshape(-1, 3)[valid] / 255.0

def apply_transform(points, T):
    homo_points = np.hstack([points, np.ones((len(points), 1))])
    return (T @ homo_points.T).T[:, :3]

def matrix_to_quaternion(R):
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    return np.array([
        (R[2,1]-R[1,2])/(4*qw),
        (R[0,2]-R[2,0])/(4*qw),
        (R[1,0]-R[0,1])/(4*qw),
        qw
    ])
    
def random_color_v2(value, maximum=255):
    """生成基于归一化值的颜色映射"""
    jet_cmap = get_cmap('jet')
    rgba = jet_cmap(value)
    return np.array(rgba[:3])  # 返回 RGB 分量

def visualize_bboxes_from_json(json_path, mask):
    

    """从 JSON 文件加载并可视化 3D 边界框"""
    # 初始化 Rerun

    # 1. 加载 JSON 数据
    all_centers = []
    all_sizes = []
    all_rot = []
    

    with open(json_path, 'r') as f:
        #['id', 'category', 'position', 'scale', 'R', 'corners']
        data = json.load(f)
        for i in range(len(data)):
            center = np.array(data[i]['position'])
            size = np.array(data[i]['scale'])
            # size = np.array([data[i]['scale'][1], data[i]['scale'][0], data[i]['scale'][2]])
            # print("center", center)
            rotation_matrix = np.array(data[i]['R'])
            all_centers.append(center)
            all_sizes.append(size)
            all_rot.append(rotation_matrix)
    all_centers = np.array(all_centers)
    all_sizes = np.array(all_sizes)
    all_rot = np.array(all_rot)
    
    # 2. 生成 3D  colors
    all_colors = [random_color_v2(ind/(all_centers.shape[0])) for ind in range(all_centers.shape[0])]
    
    quaternions = [
        rr.Quaternion(
            xyzw=Rotation.from_matrix(r).as_quat()
        )
        for r in all_rot
    ]

    quaternions = [ quaternions[j] for j in range(len(quaternions)) if mask[j] ]  # 仅保留可见的四元数
    all_colors = [ all_colors[j] for j in range(len(all_colors)) if mask[j] ] 
    
    # 7. 生成标签 ID
    labels = [f"obj_{i}" for i in range(all_centers.shape[0]) if mask[i] ]
    
    # 8. 使用 Rerun 可视化 [9](@ref)
    rr.log(
        "world/bboxes",
        rr.Boxes3D(
            centers=all_centers[mask],
            sizes=all_sizes[mask],
            quaternions=quaternions,
            colors=all_colors,
            labels=labels
        )
    )
    print(f"成功可视化 {np.sum(mask)} 个 3D 边界框/ {len(mask)} total")
    
def load_scene_data(scene_path):
    """加载场景数据"""

    # 加载GT instances
    instances_path = os.path.join(scene_path, 'instances.json')
    with open(instances_path, 'r') as f:
        instances_data = json.load(f)

    # 验证并提取有效的corners数据
    corners_array, valid_instances = validate_and_extract_corners(instances_data)

    # 加载相机参数和poses
    K = np.loadtxt(os.path.join(scene_path, 'K_depth.txt')).reshape(3, 3)
    K_rgb = np.loadtxt(os.path.join(scene_path, 'K_rgb.txt')).reshape(3, 3)
    poses = np.load(os.path.join(scene_path, 'all_poses.npy'))

    # 加载深度图列表并获取标准尺寸
    depth_path = os.path.join(scene_path, 'depth')
    num_images = len([f for f in os.listdir(depth_path) if f.endswith('.png')])

    # 验证poses数量与图片数量的一致性
    if len(poses) != num_images:
        print(f"Warning: Number of poses ({len(poses)}) != number of images ({num_images})")
        num_images = min(len(poses), num_images)

    # 加载GT点云

    return {
        'instances': valid_instances,  # 使用验证后的instances
        'corners_array': corners_array,  # 预处理的corners数组
        'K': K,
        'K_rgb': K_rgb,  # 假设K_rgb与K相同
        'poses': poses,
        'num_images': num_images,
        'depth_path': depth_path,
        'scene_path': scene_path
    }
    
def validate_and_extract_corners(instances_data):
    """验证并提取corners数据，确保形状一致"""
    valid_corners = []
    valid_instances = []
    invalid_count = 0

    for i, item in enumerate(instances_data):
        try:
            if "corners" not in item:
                print(f"Warning: Instance {i} missing 'corners' field")
                invalid_count += 1
                continue

            corners = np.array(item["corners"])

            # 检查corners的形状
            if corners.shape != (8, 3):
                print(f"Warning: Instance {i} has invalid corners shape {corners.shape}, expected (8, 3)")
                invalid_count += 1
                continue

            # 检查是否包含NaN或无穷值
            if np.any(np.isnan(corners)) or np.any(np.isinf(corners)):
                print(f"Warning: Instance {i} contains NaN or infinite values")
                invalid_count += 1
                continue

            valid_corners.append(corners)
            valid_instances.append(item)

        except Exception as e:
            print(f"Warning: Error processing instance {i}: {str(e)}")
            invalid_count += 1
            continue

    if invalid_count > 0:
        print(f"Filtered out {invalid_count} invalid instances, keeping {len(valid_corners)} valid instances")

    if len(valid_corners) == 0:
        raise ValueError("No valid corner data found in instances")

    # 现在所有corners都有相同的形状，可以安全地stack
    corners_array = np.stack(valid_corners, axis=0)

    return corners_array, valid_instances