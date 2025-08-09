import numpy as np
import json
import os
import random
from scipy.spatial import KDTree
import open3d as o3d
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import pickle


class SceneSamplerWithTransform:
    def __init__(self, data_root='/data/CA-1M-slam/'):
        self.data_root = data_root
        self.all_seq = [d for d in os.listdir(data_root)
                        if os.path.isdir(os.path.join(data_root, d))]

    def validate_and_extract_corners(self, instances_data):
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

    def get_depth_image_size(self, depth_path):
        """获取深度图的标准尺寸"""
        depth_files = [f for f in os.listdir(depth_path) if f.endswith('.png')]
        if not depth_files:
            raise ValueError(f"No depth images found in {depth_path}")

        # 检查前几张图片的尺寸
        sizes = []
        for i in range(min(5, len(depth_files))):
            depth_file = os.path.join(depth_path, f'{i}.png')
            if os.path.exists(depth_file):
                d = o3d.io.read_image(depth_file)
                d_array = np.asarray(d)
                sizes.append(d_array.shape)

        if not sizes:
            raise ValueError("Cannot determine depth image size")

        # 使用最常见的尺寸作为标准尺寸
        from collections import Counter
        most_common_size = Counter(sizes).most_common(1)[0][0]
        print(f"Using standard depth image size: {most_common_size}")

        return most_common_size

    def load_scene_data(self, scene_id):
        """加载场景数据"""
        scene_path = os.path.join(self.data_root, scene_id)

        # 加载GT instances
        instances_path = os.path.join(scene_path, 'instances.json')
        with open(instances_path, 'r') as f:
            instances_data = json.load(f)

        # 验证并提取有效的corners数据
        corners_array, valid_instances = self.validate_and_extract_corners(instances_data)

        # 加载相机参数和poses
        K = np.loadtxt(os.path.join(scene_path, 'K_depth.txt')).reshape(3, 3)
        poses = np.load(os.path.join(scene_path, 'all_poses.npy'))

        # 加载深度图列表并获取标准尺寸
        depth_path = os.path.join(scene_path, 'depth')
        num_images = len([f for f in os.listdir(depth_path) if f.endswith('.png')])

        # 获取深度图标准尺寸
        try:
            standard_depth_size = self.get_depth_image_size(depth_path)
        except Exception as e:
            print(f"Warning: Could not determine depth image size: {str(e)}")
            standard_depth_size = (480, 640)  # 默认尺寸

        # 验证poses数量与图片数量的一致性
        if len(poses) != num_images:
            print(f"Warning: Number of poses ({len(poses)}) != number of images ({num_images})")
            num_images = min(len(poses), num_images)

        # 加载GT点云
        gt_ply_path = os.path.join(scene_path, 'mesh.ply')

        return {
            'instances': valid_instances,  # 使用验证后的instances
            'corners_array': corners_array,  # 预处理的corners数组
            'K': K,
            'poses': poses,
            'num_images': num_images,
            'depth_path': depth_path,
            'standard_depth_size': standard_depth_size,
            'gt_ply_path': gt_ply_path,
            'scene_path': scene_path
        }

    def load_depth_maps(self, depth_path, image_indices, standard_size):
        """按需加载指定的深度图，统一尺寸"""
        depth_maps = []

        for idx in image_indices:
            depth_file = os.path.join(depth_path, f'{idx}.png')
            if not os.path.exists(depth_file):
                print(f"Warning: Depth file not found: {depth_file}, using zeros")
                # 使用零填充的深度图
                depth_maps.append(np.zeros(standard_size, dtype=np.uint16))
                continue

            try:
                d = o3d.io.read_image(depth_file)
                depth_array = np.asarray(d)

                # 检查深度图是否有效
                if depth_array.size == 0:
                    print(f"Warning: Empty depth map: {depth_file}, using zeros")
                    depth_maps.append(np.zeros(standard_size, dtype=np.uint16))
                    continue

                # 统一尺寸
                if depth_array.shape != standard_size:
                    print(
                        f"Warning: Depth image {depth_file} size {depth_array.shape} != standard size {standard_size}, resizing")
                    # 调整尺寸
                    depth_array = cv2.resize(depth_array, (standard_size[1], standard_size[0]),
                                             interpolation=cv2.INTER_NEAREST)

                depth_maps.append(depth_array)

            except Exception as e:
                print(f"Warning: Error loading depth map {depth_file}: {str(e)}, using zeros")
                depth_maps.append(np.zeros(standard_size, dtype=np.uint16))

        return np.stack(depth_maps, axis=0)

    def compute_anchor_overlap(self, scene_data, anchor_idx, candidate_indices):
        """计算所有候选图片与锚点图片的重叠度"""
        K = scene_data['K']
        poses = scene_data['poses']
        corners_array = scene_data['corners_array']  # 使用预处理的corners数组

        # 验证索引有效性
        if anchor_idx >= len(poses):
            print(f"Warning: Anchor index {anchor_idx} >= poses length {len(poses)}")
            return {}

        anchor_pose = poses[anchor_idx]
        overlaps = {}

        for candidate_idx in candidate_indices:
            if candidate_idx == anchor_idx:
                continue

            if candidate_idx >= len(poses):
                print(f"Warning: Candidate index {candidate_idx} >= poses length {len(poses)}")
                continue

            candidate_pose = poses[candidate_idx]

            try:
                # 计算候选图片与锚点的重叠度
                overlap_score = self.calculate_view_overlap(
                    corners_array, K, anchor_pose, candidate_pose
                )
                overlaps[candidate_idx] = overlap_score
            except Exception as e:
                print(f"Warning: Error calculating overlap for candidate {candidate_idx}: {str(e)}")
                overlaps[candidate_idx] = 0.0

        return overlaps

    def calculate_view_overlap(self, corners, K, pose1, pose2):
        """计算两个视图的重叠度"""
        try:
            # 将3D corners投影到两个视图
            proj1 = self.project_corners_to_image(corners, K, pose1)
            proj2 = self.project_corners_to_image(corners, K, pose2)

            # 计算有效投影的重叠
            valid1 = self.get_valid_projections(proj1)
            valid2 = self.get_valid_projections(proj2)

            # 计算交集与并集的比例
            intersection = np.logical_and(valid1, valid2)
            union = np.logical_or(valid1, valid2)

            if np.sum(union) == 0:
                return 0.0

            overlap_ratio = np.sum(intersection) / np.sum(union)
            return overlap_ratio
        except Exception as e:
            print(f"Warning: Error in calculate_view_overlap: {str(e)}")
            return 0.0

    def project_corners_to_image(self, corners, K, pose, img_size=(480, 640)):
        """将3D corners投影到图像平面"""
        N, num_corners, _ = corners.shape

        # 转换为齐次坐标
        hom_corners = np.concatenate([corners, np.ones((N, num_corners, 1))], axis=-1)

        # 变换到相机坐标系
        try:
            pose_inv = np.linalg.inv(pose)
        except np.linalg.LinAlgError:
            print("Warning: Singular pose matrix, using pseudo-inverse")
            pose_inv = np.linalg.pinv(pose)

        cam_points = np.dot(hom_corners, pose_inv.T)[..., :3]

        # 投影到图像平面
        x, y, z = cam_points[..., 0], cam_points[..., 1], cam_points[..., 2]

        # 避免除零
        z = np.where(z > 0.01, z, 0.01)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = fx * x / z + cx
        v = fy * y / z + cy

        return np.stack([u, v, z], axis=-1)

    def get_valid_projections(self, projections, img_size=(480, 640)):
        """获取有效投影的掩码"""
        u, v, z = projections[..., 0], projections[..., 1], projections[..., 2]
        H, W = img_size

        valid = ((u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.1) & (z < 100.0))

        # 对于每个bbox，如果至少有4个角点可见，则认为该bbox可见
        bbox_valid = np.sum(valid, axis=1) >= 4
        return bbox_valid

    def sample_with_overlap_constraint(self, scene_data, num_samples=50,
                                       min_images=2, max_images=32,
                                       overlap_threshold=0.3,  # 降低阈值
                                       time_diversity_factor=3,  # 时间分散
                                       max_retries=200):  # 增加重试次数
        """基于重叠约束的改进采样策略"""
        num_images = scene_data['num_images']

        # 检查是否有足够的图片
        if num_images < min_images:
            print(f"Warning: Scene has only {num_images} images, less than minimum required {min_images}")
            # 尝试生成一些简单的采样
            if num_images >= 1:
                simple_samples = []
                for i in range(min(num_samples, num_images)):
                    if num_images == 1:
                        simple_samples.append([0])
                    else:
                        # 简单的随机采样，不考虑overlap
                        sample_size = min(random.randint(1, min_images), num_images)
                        sample = random.sample(range(num_images), sample_size)
                        simple_samples.append(sorted(sample))
                return simple_samples
            return []

        samples = []
        failed_attempts = 0

        for sample_idx in range(num_samples):
            sample_found = False

            # 尝试多次找到满足条件的采样
            for retry in range(max_retries):
                # 随机选择采样图片数量
                target_sample_size = random.randint(min_images, min(max_images, num_images))

                # 随机选择锚点图像
                anchor_idx = random.randint(0, num_images - 1)

                # 寻找与锚点有足够overlap且时间分散的候选图片
                valid_candidates = self.find_valid_candidates(
                    scene_data, anchor_idx, overlap_threshold, time_diversity_factor
                )

                # 检查是否有足够的候选图片
                required_additional = target_sample_size - 1  # 减去锚点图片

                if len(valid_candidates) >= required_additional:
                    # 从候选图片中随机选择
                    selected_candidates = random.sample(valid_candidates, required_additional)
                    selected_images = [anchor_idx] + selected_candidates
                    samples.append(sorted(selected_images))
                    sample_found = True
                    break

                # 如果候选图片不够，尝试降低采样数量
                if len(valid_candidates) >= min_images - 1:  # 至少要满足最小采样数量
                    # 使用所有有效候选图片
                    selected_images = [anchor_idx] + valid_candidates
                    samples.append(sorted(selected_images))
                    sample_found = True
                    break

                # 如果仍然不够，进一步降低要求
                if len(valid_candidates) > 0 and target_sample_size > min_images:
                    target_sample_size = max(min_images, len(valid_candidates) + 1)
                    selected_candidates = valid_candidates
                    selected_images = [anchor_idx] + selected_candidates
                    samples.append(sorted(selected_images))
                    sample_found = True
                    break

            # 如果多次重试都失败，尝试简单随机采样作为后备
            if not sample_found:
                failed_attempts += 1
                # 生成简单的随机采样作为后备
                fallback_size = random.randint(min_images, min(max_images, num_images))
                fallback_sample = random.sample(range(num_images), fallback_size)
                samples.append(sorted(fallback_sample))
                print(f"Warning: Used fallback sampling for sample {sample_idx + 1}")

        if failed_attempts > 0:
            print(f"Warning: {failed_attempts}/{num_samples} samples used fallback strategy")

        print(f"Generated {len(samples)} samples")
        return samples

    def find_valid_candidates(self, scene_data, anchor_idx, overlap_threshold, time_diversity_factor):
        """找到与锚点有足够overlap且时间分散的候选图片"""
        num_images = scene_data['num_images']

        # 首先应用时间分散性约束，排除连续帧
        time_diverse_candidates = []
        for i in range(num_images):
            if abs(i - anchor_idx) >= time_diversity_factor:
                time_diverse_candidates.append(i)

        # 如果时间分散的候选图片太少，逐步放宽约束
        min_time_gap = time_diversity_factor
        while len(time_diverse_candidates) < 5 and min_time_gap > 1:
            min_time_gap = max(1, min_time_gap - 1)
            time_diverse_candidates = []
            for i in range(num_images):
                if abs(i - anchor_idx) >= min_time_gap:
                    time_diverse_candidates.append(i)

        # 如果仍然不够，包含所有非锚点图片
        if len(time_diverse_candidates) < 3:
            time_diverse_candidates = [i for i in range(num_images) if i != anchor_idx]

        if len(time_diverse_candidates) == 0:
            return []

        # 计算与锚点的重叠度
        try:
            overlaps = self.compute_anchor_overlap(scene_data, anchor_idx, time_diverse_candidates)
        except Exception as e:
            print(f"Warning: Error computing overlaps for anchor {anchor_idx}: {str(e)}")
            # 如果overlap计算失败，返回时间分散的候选图片
            return time_diverse_candidates[:min(10, len(time_diverse_candidates))]

        # 筛选出满足overlap阈值的候选图片
        valid_candidates = []
        for candidate_idx, overlap_score in overlaps.items():
            if overlap_score >= overlap_threshold:
                valid_candidates.append(candidate_idx)

        # 如果没有满足overlap阈值的候选，降低阈值重试
        if len(valid_candidates) == 0 and overlap_threshold > 0.05:
            lower_threshold = max(0.05, overlap_threshold * 0.5)
            for candidate_idx, overlap_score in overlaps.items():
                if overlap_score >= lower_threshold:
                    valid_candidates.append(candidate_idx)

        # 如果仍然没有，返回一些时间分散的候选图片
        if len(valid_candidates) == 0:
            valid_candidates = time_diverse_candidates[:min(5, len(time_diverse_candidates))]

        return valid_candidates

    def filter_gt_boxes_for_images(self, scene_data, image_indices):
        """为指定图像过滤GT boxes"""
        instances = scene_data['instances']
        corners_array = scene_data['corners_array']
        K = scene_data['K']

        # 验证image_indices的有效性
        valid_indices = [idx for idx in image_indices if idx < len(scene_data['poses'])]
        if len(valid_indices) != len(image_indices):
            print(
                f"Warning: Some image indices are invalid, using {len(valid_indices)}/{len(image_indices)} valid indices")

        if len(valid_indices) == 0:
            print("Warning: No valid image indices")
            return []

        try:
            poses = scene_data['poses'][valid_indices]
            depth_maps = self.load_depth_maps(
                scene_data['depth_path'],
                valid_indices,
                scene_data['standard_depth_size']
            )
            gt_ply_path = scene_data['gt_ply_path']

            # 提取bbox_ids
            bbox_ids = []
            for item in instances:
                # 假设instances.json中有'id'字段，如果没有则使用索引
                if 'id' in item:
                    bbox_ids.append(item['id'])
                elif 'bbox_id' in item:
                    bbox_ids.append(item['bbox_id'])
                else:
                    # 生成唯一ID
                    bbox_ids.append(f"bbox_{len(bbox_ids)}")

            # 使用filter_gt_boxes.py的逻辑
            filtered_corners, bbox_mask = self.filter_3d_corners(
                corners_array, K, poses, depth_maps, gt_ply_path
            )

            # 获取过滤后的bbox_ids
            filtered_bbox_ids = [bbox_ids[i] for i in range(len(bbox_ids)) if bbox_mask[i]]

            return filtered_bbox_ids

        except Exception as e:
            print(f"Warning: Error in filter_gt_boxes_for_images: {str(e)}")
            # 返回所有bbox_ids作为后备
            bbox_ids = []
            for item in instances:
                if 'id' in item:
                    bbox_ids.append(item['id'])
                elif 'bbox_id' in item:
                    bbox_ids.append(item['bbox_id'])
                else:
                    bbox_ids.append(f"bbox_{len(bbox_ids)}")
            return bbox_ids

    def filter_3d_corners(self, corners, K, poses, depth_maps, gt_ply_path,
                          near=0.1, far=100.0, dist_threshold=0.5):  # 放宽距离阈值
        """基于filter_gt_boxes.py的过滤逻辑"""
        try:
            # 第一步：视锥体剔除
            bbox_frustum_mask = self.frustum_culling_bbox_level(
                corners, K, poses, depth_maps, near, far
            )
            visible_bboxes = corners[bbox_frustum_mask]

            if len(visible_bboxes) == 0:
                # 如果没有可见的boxes，返回空结果
                return np.empty((0, 8, 3)), np.zeros(len(corners), dtype=bool)

            # 第二步：点云邻近性验证
            gt_points = self.load_gt_point_cloud(gt_ply_path)
            bbox_proximity_mask = self.check_bbox_proximity(
                visible_bboxes, gt_points, dist_threshold
            )

            # 组合掩码
            final_mask = np.zeros(len(corners), dtype=bool)
            final_mask[bbox_frustum_mask] = bbox_proximity_mask

            filtered_corners = corners[final_mask]
            return filtered_corners, final_mask
        except Exception as e:
            print(f"Warning: Error in filter_3d_corners: {str(e)}")
            # 返回所有corners作为后备
            return corners, np.ones(len(corners), dtype=bool)

    def frustum_culling_bbox_level(self, corners, K, poses, depth_maps, near, far):
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
        bbox_visible = count_visible >= 6

        return bbox_visible

    def load_gt_point_cloud(self, ply_path):
        """加载GT点云"""
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"GT point cloud file not found: {ply_path}")

        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)

        if len(points) == 0:
            raise ValueError(f"Empty point cloud: {ply_path}")

        return points

    def check_bbox_proximity(self, bboxes, gt_points, threshold=0.5):
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

        return proximity_mask

    def transform_to_first_frame_coordinate(self, scene_data, sample_data, scene_id, sample_idx):
        """
        将采样的图像和bbox坐标变换到第一帧图像的相机坐标系

        Args:
            scene_data: 场景数据
            sample_data: 单次采样结果，包含image_id和bbox_id
            scene_id: 场景ID
            sample_idx: 采样索引

        Returns:
            transformed_sample: 按指定格式的变换结果global output_dir
        """
        image_indices = sample_data['image_id']
        bbox_ids = sample_data['bbox_id']

        # 创建sceneid_exampleid
        sceneid_exampleid = f"{scene_id}_{sample_idx}"

        if len(image_indices) == 0:
            return {
                "sceneid_exampleid": sceneid_exampleid,
                "image_id": [],
                "pose": [],
                "bbox_id": [],
                "position": [],
                "R": [],
                "corners": []
            }

        # 找到第一帧图像（命名序号最小的图像）
        first_frame_idx = min(image_indices)
        first_frame_pose = scene_data['poses'][first_frame_idx]

        # 计算变换矩阵
        try:
            first_frame_pose_inv = np.linalg.inv(first_frame_pose)
        except np.linalg.LinAlgError:
            print("Warning: Singular first frame pose matrix, using pseudo-inverse")
            first_frame_pose_inv = np.linalg.pinv(first_frame_pose)

        # 计算每帧相对于第一帧的变换，按image_id顺序排列
        pose_transforms = []
        for img_idx in image_indices:
            if img_idx == first_frame_idx:
                # 第一帧自身的变换是单位矩阵
                pose_transforms.append(np.eye(4).tolist())
            else:
                # 计算从当前帧到第一帧的变换
                current_pose = scene_data['poses'][img_idx]
                relative_pose = first_frame_pose_inv @ current_pose
                pose_transforms.append(relative_pose.tolist())

        # 变换bbox坐标到第一帧坐标系
        instances = scene_data['instances']

        # 创建bbox_id到实例的映射
        bbox_id_to_instance = {}
        for instance in instances:
            if 'id' in instance:
                bbox_id_to_instance[instance['id']] = instance
            elif 'bbox_id' in instance:
                bbox_id_to_instance[instance['bbox_id']] = instance

        # 按bbox_id顺序提取变换后的数据
        transformed_positions = []
        transformed_rotations = []
        transformed_corners_list = []

        for bbox_id in bbox_ids:
            if bbox_id not in bbox_id_to_instance:
                print(f"Warning: bbox_id {bbox_id} not found in instances")
                # 使用默认值
                transformed_positions.append([0.0, 0.0, 0.0])
                transformed_rotations.append([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                transformed_corners_list.append([[0, 0, 0]] * 8)
                continue

            instance = bbox_id_to_instance[bbox_id]

            # 提取原始bbox信息
            original_corners = np.array(instance.get('corners', []))
            original_position = np.array(instance.get('position', [0, 0, 0]))

            # 处理旋转矩阵
            if 'R' in instance:
                original_R = np.array(instance['R'])
            elif 'rotation' in instance:
                original_R = np.array(instance['rotation'])
            else:
                original_R = np.eye(3)  # 默认无旋转

            # 变换corners到第一帧坐标系
            if len(original_corners) > 0 and original_corners.shape == (8, 3):
                # 转换为齐次坐标
                corners_homogeneous = np.concatenate([
                    original_corners,
                    np.ones((original_corners.shape[0], 1))
                ], axis=1)

                # 变换到第一帧坐标系
                transformed_corners_hom = (first_frame_pose_inv @ corners_homogeneous.T).T
                transformed_corners = transformed_corners_hom[:, :3]
                transformed_corners_list.append(transformed_corners.tolist())
            else:
                # 使用默认值或从position和R计算
                transformed_corners_list.append([[0, 0, 0]] * 8)

            # 变换position到第一帧坐标系
            if len(original_position) == 3:
                position_homogeneous = np.array([*original_position, 1])
                transformed_position_hom = first_frame_pose_inv @ position_homogeneous
                transformed_position = transformed_position_hom[:3]
                transformed_positions.append(transformed_position.tolist())
            else:
                transformed_positions.append([0.0, 0.0, 0.0])

            # 变换旋转矩阵到第一帧坐标系
            if original_R.shape == (3, 3):
                transformed_R = first_frame_pose_inv[:3, :3] @ original_R
                transformed_rotations.append(transformed_R.tolist())
            else:
                transformed_rotations.append([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # 构建按指定格式的结果
        result = {
            "sceneid_exampleid": sceneid_exampleid,
            "image_id": image_indices,
            "pose": pose_transforms,
            "bbox_id": bbox_ids,
            "position": transformed_positions,
            "R": transformed_rotations,
            "corners": transformed_corners_list
        }

        return result

    def process_scene(self, scene_id):
        """处理单个场景"""
        print(f"Processing scene: {scene_id}")

        try:
            # 加载场景数据
            scene_data = self.load_scene_data(scene_id)
            print(
                f"  Loaded scene data: {scene_data['num_images']} images, {len(scene_data['corners_array'])} instances")

            # 进行50次采样
            samples = self.sample_with_overlap_constraint(scene_data, min_images=3, max_images=3)

            print(f"  Generated {len(samples)} valid samples")

            if len(samples) == 0:
                print(f"  Warning: No valid samples generated for scene {scene_id}")
                # 创建一个空的结果文件
                output_dir = self.data_root #'/data/CA-1M-sample-transformed/'
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'{scene_id}_sampling_results_transformed.json')
                with open(output_path, 'w') as f:
                    json.dump([], f, indent=2)
                return []

            # 处理每次采样
            results = []
            for sample_idx, image_indices in enumerate(samples):
                print(f"  Processing sample {sample_idx + 1}/{len(samples)} with {len(image_indices)} images")

                try:
                    # 为当前采样的图像过滤GT boxes
                    filtered_bbox_ids = self.filter_gt_boxes_for_images(
                        scene_data, image_indices
                    )

                    # 创建基础采样结果
                    sample_data = {
                        "image_id": image_indices,
                        "bbox_id": filtered_bbox_ids
                    }

                    # 执行坐标变换到第一帧坐标系（添加scene_id和sample_idx参数）
                    transformed_sample = self.transform_to_first_frame_coordinate(
                        scene_data, sample_data, scene_id, sample_idx
                    )

                    results.append(transformed_sample)

                    print(
                        f"    Sample {sample_idx + 1}: {len(image_indices)} images -> {len(filtered_bbox_ids)} bbox_ids (ID: {transformed_sample['sceneid_exampleid']})")

                except Exception as e:
                    print(f"  Warning: Error processing sample {sample_idx + 1}: {str(e)}")
                    # 添加一个空的结果而不是跳过
                    results.append({
                        "sceneid_exampleid": f"{scene_id}_{sample_idx}",
                        "image_id": image_indices,
                        "pose": [],
                        "bbox_id": [],
                        "position": [],
                        "R": [],
                        "corners": []
                    })
                    continue

            # 保存结果
            output_dir = self.data_root #'/data/CA-1M-sample-transformed/'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, scene_id, f'{scene_id}_sampling_results_transformed.json')

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"  Saved {len(results)} transformed results to: {output_path}")
            return results

        except Exception as e:
            print(f"Error processing scene {scene_id}: {str(e)}")
            # 即使出错也创建一个空的结果文件
            try:
                output_dir = '/data/CA-1M-sample-transformed/'
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'{scene_id}_sampling_results_transformed.json')
                with open(output_path, 'w') as f:
                    json.dump([], f, indent=2)
                print(f"  Created empty result file: {output_path}")
            except Exception as save_error:
                print(f"  Failed to create empty result file: {str(save_error)}")
            return []

    def run_all_scenes(self, max_workers=None):
        """处理所有场景"""
        print(f"Starting to process {len(self.all_seq)} scenes...")

        if max_workers is None:
            max_workers = min(4, mp.cpu_count())

        if max_workers > 1:
            print(f"Using multiprocessing with {max_workers} workers")
            with mp.Pool(max_workers) as pool:
                results = list(tqdm(
                    pool.imap(self.process_scene, self.all_seq),
                    total=len(self.all_seq),
                    desc="Processing scenes"
                ))
        else:
            print("Using single process")
            results = []
            for scene_id in tqdm(self.all_seq, desc="Processing scenes"):
                result = self.process_scene(scene_id)
                results.append(result)

        print("All scenes processed!")
        return results


def main():
    sampler = SceneSamplerWithTransform(data_root='/media/lyq/temp/dataset/train-CA-1M-slam/')

    # 处理单个场景:
    # scene_id = "42444750"
    # result = sampler.process_scene(scene_id)
    # print(result)
    # 处理所有场景:
    results = sampler.run_all_scenes(max_workers=8)

    print("Processing completed!")


if __name__ == "__main__":
    main()