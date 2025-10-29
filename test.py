import numpy as np
import torch
import os 
import json
import cv2
from scipy.spatial import KDTree


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _rotation_align_a_to_b(a, b):
    """
    返回将向量 a 旋到向量 b 的 3x3 旋转矩阵（右手系）。
    a,b 为长度为3的一维向量。
    """
    a = _normalize(a)
    b = _normalize(b)
    c = np.dot(a, b)            # cos(theta)
    if c >= 1.0 - 1e-12:        # 已对齐
        return np.eye(3)
    if c <= -1.0 + 1e-12:       # 反向（180°），任选一条与 a 正交的轴
        # 找到任意与 a 不平行的基向量
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = _normalize(np.cross(a, tmp))
        # 罗德里格斯公式，theta=pi
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + 2 * K @ K  # 因为 sin(pi)=0, (1-cos(pi))=2
        return R

    v = np.cross(a, b)          # 旋转轴方向（未归一化）
    s = np.linalg.norm(v)       # sin(theta)
    v = v / s                   # 归一化轴
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    # 罗德里格斯公式：R = I + K*sin + K^2*(1-cos)
    R = np.eye(3) + K * s + (K @ K) * (1 - c)
    return R

def upright_boxes(corners: np.ndarray, tilt_deg_thresh: float = 5.0):
    """
    输入:
        corners: [N, 8, 3] 的 numpy 数组
        tilt_deg_thresh: 判定“非垂直”的阈值（度）
    返回:
        upright_corners: [N, 8, 3]，将“倾斜”的 box 旋正后的角点；其余保持不变
        tilted_mask: [N] 的布尔数组，True 表示该 box 原本非垂直，被处理过
        tilt_degs: [N] 与世界 z 轴的夹角（度）
    """
    assert corners.ndim == 3 and corners.shape[1:] == (8, 3), "corners 形状需为 [N, 8, 3]"
    N = corners.shape[0]
    z_hat = np.array([0.0, 0.0, 1.0])

    upright = np.empty_like(corners)
    tilted_mask = np.zeros((N,), dtype=bool)
    tilt_degs = np.zeros((N,), dtype=float)

    for i in range(N):
        pts = corners[i]                       # [8,3]
        c = pts.mean(axis=0)                   # 质心
        X = pts - c                            # 去中心化 [8,3]

        # 用 SVD/PCA 求主轴（列为主方向）
        # X = U S V^T，主方向为 V 的列（或 V^T 的行）
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        axes = Vt.T                             # [3,3], 每列一个主方向，按方差大小降序

        # 选与世界 z 轴最接近的主轴，视为 box 的“局部 z 轴”
        dots = np.abs(axes.T @ z_hat)          # 与 z 的投影绝对值，[3]
        k = int(np.argmax(dots))
        local_z = axes[:, k]                   # 与 z 最近的轴方向（单位向量）

        # 计算倾斜角
        cosang = np.clip(np.abs(np.dot(_normalize(local_z), z_hat)), 0.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        tilt_degs[i] = float(ang)

        if ang > tilt_deg_thresh:
            # 旋转到 z 轴对齐（绕质心）
            R = _rotation_align_a_to_b(local_z, z_hat)
            X_rot = (R @ X.T).T                # [8,3]
            pts_upright = X_rot + c
            upright[i] = pts_upright
            tilted_mask[i] = True
        else:
            # 已足够垂直，直接保留
            upright[i] = pts

    return upright, tilted_mask, tilt_degs




def apply_transform(points, T):
    homo_points = np.hstack([points, np.ones((len(points), 1))])
    return (T @ homo_points.T).T[:, :3]

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



def load_scene_data(scene_id, CA1M_DIR):
    """加载场景数据"""
    scene_path = os.path.join(CA1M_DIR, scene_id)
    # 加载GT instances
    instances_path = os.path.join(scene_path, 'instances.json')
    print('path:', instances_path)
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

def filter_gt_boxes_for_images(scene_data, image_indices):
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
        filtered_corners = filter_3d_corners(
            corners_array, K, poses, depth_maps,
            frustum_threshold=4, dist_threshold=0.2 #TODO:0.5 0.2
        )

        return filtered_corners

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

def depth_to_pointcloud(depth_maps, K, T_wc):
    N_imgs = depth_maps.shape[0]
    all_pts = []
    for i in range(N_imgs):
        depth = depth_maps[i]
            
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        height, width = depth.shape
        
        # 生成网格坐标
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth.flatten()
        x = (u.flatten() - cx) * z / fx
        y = (v.flatten() - cy) * z / fy
        
        # 转换到世界坐标系
        points_cam = np.vstack([x, y, z]).T
        points_world = apply_transform(points_cam, T_wc[i])
        
        # 过滤无效点并提取颜色
        valid = z > 0
        all_pts.append(points_world[valid])
        
    all_pts = np.concatenate(all_pts, axis=0)
    
    return all_pts


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
        # print(self.seqname,"邻近性检查通过", proximity_mask.sum(), '/', len(proximity_mask))
        return proximity_mask

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

            valid_corner = valid_z & valid_uv
            bbox_mask |= valid_corner
            
        except Exception as e:
            print(f"Warning: Error processing frame {i}: {str(e)}")
            continue

    # 至少4个角点可见
    count_visible = np.sum(bbox_mask, axis=1)
    bbox_visible = count_visible >= threshold
    # print(self.seqname,"视锥保留", bbox_visible.sum(),'/', len(bbox_visible))
    return bbox_visible


def filter_3d_corners(corners, K, poses, depth_maps,
                          near=0.1, far=100.0, frustum_threshold=4, dist_threshold=0.5):  # 放宽距离阈值
    """基于filter_gt_boxes.py的过滤逻辑"""
    try:
        # 第一步：视锥体剔除
        bbox_frustum_mask = frustum_culling_bbox_level(
            corners, K, poses, depth_maps, near, far,
            threshold=frustum_threshold
        )   
        
        visible_bboxes = corners[bbox_frustum_mask]

        if len(visible_bboxes) == 0:
            # 如果没有可见的boxes，返回空结果
            return np.empty((0, 8, 3)), np.zeros(len(corners), dtype=bool)

        # 第二步：点云邻近性验证
        #TODO:根据场景整体点云，裁剪出相机可视部分的点云，否则会错误保留被遮挡的那些box,CA1M数据集可以直接使用depth，因为它的depth是根据Mesh渲染的
        gt_points = depth_to_pointcloud(depth_maps, K, poses)

        bbox_proximity_mask = check_bbox_proximity(
            visible_bboxes, gt_points, dist_threshold
        )
        #TODO: test
        # bbox_proximity_mask[:] = True
        # 组合掩码
        final_mask = np.zeros(len(corners), dtype=bool)
        final_mask[bbox_frustum_mask] = bbox_proximity_mask

        filtered_corners = corners[final_mask]
        final_mask_frameid = [i for i in range(final_mask.shape[0]) if final_mask[i]]
        # print(self.seqname, "过滤后保留的GT boxes数量:", len(filtered_corners), "对应的box ID:", final_mask_frameid)
        return filtered_corners
    
    except Exception as e:
        print(f"Warning: Error in filter_3d_corners: {str(e)}")
        # 返回所有corners作为后备
        return corners


seq_name= '42444750'
data_dir = '/data/lyq/ca1m/ca1m/train-CA-1M-slam/'
image_idxs = [180, 200]
# Load sequence data
scene_data = load_scene_data(seq_name, data_dir)
# Load Box GT information
filtered_bbox_corners = filter_gt_boxes_for_images(scene_data, image_idxs)
print("filtered_bbox_corners:", filtered_bbox_corners.shape)


# ------------------ 用法示例 ------------------
# boxes: np.ndarray 形如 [N,8,3]
upright, tilted_mask, tilt_degs = upright_boxes(filtered_bbox_corners, tilt_deg_thresh=5.0)
# upright 即为处理后的 [N,8,3]；tilted_mask 标出哪些被旋正；tilt_degs 给出每个 box 的倾斜角。
print("upright boxes shape:", upright.shape)
print('tilted_mask', tilted_mask)
np.save('/home/lanyuqing/myproject/vggt/vis_results/gt_box.npy',filtered_bbox_corners[tilted_mask])
np.save('/home/lanyuqing/myproject/vggt/vis_results/upright_boxes.npy', upright[tilted_mask])
