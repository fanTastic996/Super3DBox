import os
import json
import numpy as np
import cv2
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
import argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import uuid
import open3d as o3d


class TransformedSampleVisualizer:
    def __init__(self, scene_path, transformed_json_path):
        """
        初始化变换采样数据可视化器

        Args:
            scene_path (str): 场景数据路径
            transformed_json_path (str): 变换后的采样JSON文件路径
        """
        self.scene_path = scene_path
        self.transformed_json_path = transformed_json_path

        # 颜色管理，保持实例颜色一致性
        self.id_to_color = {}

        # 加载数据
        self.load_scene_data()
        self.load_transformed_sample_data()

    def load_scene_data(self):
        """加载场景数据"""
        print("加载场景数据...")

        # 加载instances.json（用于获取原始instances信息）
        instances_path = os.path.join(self.scene_path, 'instances.json')
        with open(instances_path, 'r') as f:
            self.instances_data = json.load(f)

        # 加载原始poses（用于变换点云）
        poses_path = os.path.join(self.scene_path, 'all_poses.npy')
        if os.path.exists(poses_path):
            self.original_poses = np.load(poses_path)
        else:
            self.original_poses = None

        # 加载mesh.ply点云
        mesh_path = os.path.join(self.scene_path, 'mesh.ply')
        if os.path.exists(mesh_path):
            self.original_gt_pointcloud = self.load_gt_point_cloud(mesh_path)
            print(f"加载了GT点云: {len(self.original_gt_pointcloud)} 个点")
        else:
            self.original_gt_pointcloud = None
            print("警告: 未找到mesh.ply文件")

        # 加载重力对齐变换
        gravity_path = os.path.join(self.scene_path, 'T_gravity.npy')
        if os.path.exists(gravity_path):
            self.T_gravity = np.load(gravity_path)
            print("加载了重力对齐变换矩阵")
        else:
            self.T_gravity = np.eye(4)
            print("使用单位矩阵作为重力对齐变换")

        # 加载相机内参
        k_rgb_path = os.path.join(self.scene_path, 'K_rgb.txt')
        if os.path.exists(k_rgb_path):
            self.K_rgb = np.loadtxt(k_rgb_path).reshape(3, 3)
        else:
            k_depth_path = os.path.join(self.scene_path, 'K_depth.txt')
            if os.path.exists(k_depth_path):
                self.K_rgb = np.loadtxt(k_depth_path).reshape(3, 3)
            else:
                self.K_rgb = None

        # RGB图片路径
        self.rgb_path = os.path.join(self.scene_path, 'rgb')

    def load_gt_point_cloud(self, ply_path):
        """
        加载GT点云

        Args:
            ply_path: PLY文件路径

        Returns:
            points: 点云数组 [N, 3] 或 [N, 6] (包含颜色)
        """
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)

            # 检查是否有颜色信息
            if pcd.colors:
                colors = np.asarray(pcd.colors)
                # 合并点坐标和颜色 [N, 6] (xyz + rgb)
                points_with_colors = np.concatenate([points, colors], axis=1)
                return points_with_colors
            else:
                # 如果没有颜色，只返回坐标
                return points

        except Exception as e:
            print(f"错误: 无法加载点云文件 {ply_path}: {str(e)}")
            return None

    def load_transformed_sample_data(self):
        """加载变换后的采样数据"""
        print("加载变换后的采样数据...")

        with open(self.transformed_json_path, 'r') as f:
            self.samples = json.load(f)

        print(f"加载了 {len(self.samples)} 个变换采样")

    def transform_pointcloud_to_first_frame(self, sample):
        """
        将GT点云从全局世界坐标系变换到第一帧相机坐标系

        Args:
            sample: 采样数据，包含image_id和pose信息

        Returns:
            transformed_pointcloud: 变换后的点云
        """
        if self.original_gt_pointcloud is None:
            return None

        image_ids = sample.get('image_id', [])
        if len(image_ids) == 0:
            return None

        # 获取第一帧的图像ID
        first_frame_id = min(image_ids)

        if self.original_poses is None or first_frame_id >= len(self.original_poses):
            print("警告: 无法获取第一帧的原始pose")
            return self.original_gt_pointcloud

        # 获取第一帧在原始坐标系中的pose
        first_frame_pose = self.original_poses[first_frame_id]

        # 计算变换矩阵（从世界坐标系到第一帧相机坐标系）
        try:
            first_frame_pose_inv = np.linalg.inv(first_frame_pose)
        except np.linalg.LinAlgError:
            print("警告: 第一帧pose矩阵奇异，使用伪逆")
            first_frame_pose_inv = np.linalg.pinv(first_frame_pose)

        # 提取点云坐标
        if self.original_gt_pointcloud.shape[1] == 6:  # 包含颜色信息
            points = self.original_gt_pointcloud[:, :3]
            colors = self.original_gt_pointcloud[:, 3:6]
        else:  # 只有坐标
            points = self.original_gt_pointcloud
            colors = None

        # 转换为齐次坐标
        points_homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

        # 变换到第一帧相机坐标系
        transformed_points_hom = (first_frame_pose_inv @ points_homogeneous.T).T
        transformed_points = transformed_points_hom[:, :3]

        # 重新组合点云数据
        if colors is not None:
            transformed_pointcloud = np.concatenate([transformed_points, colors], axis=1)
        else:
            transformed_pointcloud = transformed_points

        return transformed_pointcloud

    def random_color(self, rgb=True):
        """生成随机颜色"""
        color = np.random.randint(50, 256, 3)
        if rgb:
            return color.tolist()
        return color / 255.0

    def get_instance_color(self, instance_id):
        """获取实例的一致颜色"""
        if instance_id not in self.id_to_color:
            self.id_to_color[instance_id] = self.random_color(rgb=True)
        return self.id_to_color[instance_id]

    def log_transformed_instances(self, sample, prefix="sample", **kwargs):
        """
        记录变换后的3D实例到rerun

        Args:
            sample: 采样数据，包含变换后的bbox信息
            prefix: 记录路径前缀
        """
        bbox_ids = sample.get('bbox_id', [])
        positions = sample.get('position', [])
        rotations = sample.get('R', [])
        corners_list = sample.get('corners', [])

        if not bbox_ids:
            return

        centers = []
        sizes = []
        quaternions = []
        colors = []
        labels = []

        for i, bbox_id in enumerate(bbox_ids):
            try:
                # 使用变换后的position作为中心
                if i < len(positions):
                    center = np.array(positions[i])
                else:
                    print(f"警告: bbox {bbox_id} 缺少position信息")
                    continue

                # 使用变换后的corners计算size
                if i < len(corners_list) and len(corners_list[i]) == 8:
                    corners = np.array(corners_list[i])
                    # 计算尺寸
                    min_coords = np.min(corners, axis=0)
                    max_coords = np.max(corners, axis=0)
                    size = max_coords - min_coords
                else:
                    # 如果没有corners，使用默认尺寸
                    size = np.array([1.0, 1.0, 1.0])

                # 使用变换后的旋转矩阵
                if i < len(rotations):
                    R = np.array(rotations[i])
                    if R.shape == (3, 3):
                        # 将旋转矩阵转换为四元数
                        rotation = Rotation.from_matrix(R)
                        quaternion = rr.Quaternion(xyzw=rotation.as_quat())
                    else:
                        quaternion = rr.Quaternion(xyzw=[0, 0, 0, 1])
                else:
                    quaternion = rr.Quaternion(xyzw=[0, 0, 0, 1])

                centers.append(center)
                sizes.append(size)
                quaternions.append(quaternion)
                colors.append(self.get_instance_color(bbox_id))
                labels.append(str(bbox_id))

            except Exception as e:
                print(f"警告: 处理bbox {bbox_id} 时出错: {str(e)}")
                continue

        if centers:
            # 使用rerun.Boxes3D记录3D边界框
            rr.log(
                f"first_frame_world/{prefix}/instances",
                rr.Boxes3D(
                    centers=np.array(centers),
                    sizes=np.array(sizes),
                    quaternions=quaternions,
                    colors=colors,
                    labels=labels,
                    show_labels=False
                ),
                **kwargs
            )

    def log_transformed_pointcloud(self, sample, prefix="pointcloud", **kwargs):
        """
        记录变换后的GT点云到rerun

        Args:
            sample: 采样数据
            prefix: 记录路径前缀
        """
        transformed_pointcloud = self.transform_pointcloud_to_first_frame(sample)

        if transformed_pointcloud is None:
            return

        # 检查点云格式
        if transformed_pointcloud.shape[1] == 6:  # 包含颜色信息 [N, 6] (xyz + rgb)
            positions = transformed_pointcloud[:, :3]
            colors = transformed_pointcloud[:, 3:6]
        elif transformed_pointcloud.shape[1] == 3:  # 只有坐标 [N, 3]
            positions = transformed_pointcloud
            colors = None
        else:
            print(f"警告: 不支持的点云格式，形状为 {transformed_pointcloud.shape}")
            return

        # 下采样以提高性能
        max_points = int(5e4)  # 限制点数
        if len(positions) > max_points:
            indices = np.random.choice(len(positions), max_points, replace=False)
            positions = positions[indices]
            if colors is not None:
                colors = colors[indices]

        # 记录点云
        rr.log(
            f"first_frame_world/{prefix}",
            rr.Points3D(
                positions=positions,
                colors=colors,
                radii=0.01  # 点的半径
            ),
            **kwargs
        )

        print(f"记录了 {len(positions)} 个变换后的点云点")

    def log_camera_and_image(self, image_id, pose_matrix, sample_idx, camera_idx):
        """
        记录相机位姿和图片（在第一帧坐标系下）

        Args:
            image_id: 图片ID
            pose_matrix: 相对于第一帧的变换矩阵 [4, 4]
            sample_idx: 采样索引
            camera_idx: 相机索引
        """
        # 加载RGB图片
        img_path = os.path.join(self.rgb_path, f"{image_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.rgb_path, f"{image_id:06d}.png")

        if not os.path.exists(img_path):
            print(f"警告: 图片文件不存在: {img_path}")
            return

        try:
            image = cv2.imread(img_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image_rgb.shape[:2]

                # 记录图片
                rr.log(
                    f"device/camera_{camera_idx}/image",
                    rr.Image(image_rgb).compress()
                )

                # 记录相机参数
                if self.K_rgb is not None:
                    camera = rr.Pinhole(
                        image_from_camera=self.K_rgb,
                        resolution=[width, height]
                    )
                    rr.log(f"device/camera_{camera_idx}/image", camera)

                # 记录相机位姿到第一帧坐标系
                pose_matrix = np.array(pose_matrix)

                # 创建变换（相对于第一帧坐标系）
                pose_transform = rr.Transform3D(
                    translation=pose_matrix[:3, 3],
                    rotation=rr.Quaternion(xyzw=Rotation.from_matrix(pose_matrix[:3, :3]).as_quat())
                )

                rr.log(f"first_frame_world/camera_{camera_idx}", pose_transform)
                if self.K_rgb is not None:
                    rr.log(f"first_frame_world/camera_{camera_idx}", camera)
                    # 在第一帧坐标系中显示半透明图片
                    rr.log(f"first_frame_world/camera_{camera_idx}/image", rr.Image(image_rgb, opacity=0.5))

        except Exception as e:
            print(f"错误: 无法加载图片 {img_path}: {str(e)}")

    def visualize_sample(self, sample_idx):
        """
        可视化单个采样（在第一帧坐标系下）

        Args:
            sample_idx: 采样索引
        """
        if sample_idx >= len(self.samples):
            print(f"采样索引 {sample_idx} 超出范围")
            return

        sample = self.samples[sample_idx]
        sceneid_exampleid = sample.get("sceneid_exampleid", f"unknown_{sample_idx}")
        image_ids = sample.get("image_id", [])
        bbox_ids = sample.get("bbox_id", [])
        poses = sample.get("pose", [])

        print(f"\n=== 可视化变换采样 {sample_idx} ===")
        print(f"采样ID: {sceneid_exampleid}")
        print(f"图片数量: {len(image_ids)}")
        print(f"边界框数量: {len(bbox_ids)}")
        print(f"图片IDs: {image_ids}")
        print(f"边界框IDs: {bbox_ids}")

        # 设置时间线
        rr.set_time_sequence("sample", sample_idx)

        # 记录变换后的GT点云（静态）
        self.log_transformed_pointcloud(sample, static=True)

        # 记录变换后的3D边界框实例
        self.log_transformed_instances(sample, f"sample_{sample_idx}")

        # 记录每张图片和对应的相机（在第一帧坐标系中）
        for cam_idx, image_id in enumerate(image_ids):
            if cam_idx < len(poses):
                pose_matrix = poses[cam_idx]
                self.log_camera_and_image(image_id, pose_matrix, sample_idx, cam_idx)
            else:
                print(f"警告: 相机 {cam_idx} 缺少pose信息")

        # 记录采样信息
        rr.log(
            f"info/sample_{sample_idx}",
            rr.TextLog(f"Transformed Sample {sceneid_exampleid}: {len(image_ids)} images, {len(bbox_ids)} boxes")
        )

    def setup_blueprint(self):
        """
        设置rerun的布局蓝图
        """
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                contents=[
                    rrb.Spatial3DView(
                        name="First Frame World",
                        origin="/first_frame_world",
                        contents=[
                            "+ $origin/**",
                        ]
                    ),
                    rrb.Horizontal(
                        contents=[
                            rrb.Spatial2DView(
                                name="Image 1",
                                origin="/device/camera_0/image",
                                contents=[
                                    "+ $origin/**",
                                ]
                            ),
                            rrb.Spatial2DView(
                                name="Image 2",
                                origin="/device/camera_1/image",
                                contents=[
                                    "+ $origin/**",
                                ]
                            ),
                            rrb.Spatial2DView(
                                name="Image 3",
                                origin="/device/camera_2/image",
                                contents=[
                                    "+ $origin/**",
                                ]
                            ),
                            rrb.TextLogView(
                                name="Info",
                                origin="/info",
                            ),
                        ],
                        name="Cameras"
                    )
                ]
            )
        )
        return blueprint

    def visualize_all_samples(self):
        """可视化所有采样"""
        print("开始可视化所有变换采样...")

        # 设置第一帧坐标系（右手坐标系）
        rr.log("/first_frame_world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        for sample_idx in range(len(self.samples)):
            self.visualize_sample(sample_idx)

        print("可视化完成！")


def visualize_transformed_scene_samples(scene_path, transformed_json_path, sample_indices=None, app_name=None):
    """
    便捷函数：可视化变换后的场景采样数据

    Args:
        scene_path (str): 场景数据路径
        transformed_json_path (str): 变换后的采样JSON文件路径
        sample_indices (list, optional): 要可视化的采样索引列表，None表示全部
        app_name (str, optional): rerun应用名称
    """
    # 获取场景名称
    scene_name = os.path.basename(scene_path)
    if app_name is None:
        app_name = f"transformed_samples_{scene_name}"

    # 初始化rerun
    recording = rr.new_recording(
        application_id=app_name,
        recording_id=uuid.uuid4(),
        make_default=True
    )

    # 创建可视化器
    visualizer = TransformedSampleVisualizer(scene_path, transformed_json_path)

    # 设置蓝图并发送
    blueprint = visualizer.setup_blueprint()
    recording.send_blueprint(blueprint, make_active=True)

    # 启动rerun查看器
    rr.spawn()

    # 记录场景信息
    rr.log("info/scene", rr.TextLog(f"Scene: {scene_name}"))
    rr.log("info/scene", rr.TextLog(f"Total transformed samples: {len(visualizer.samples)}"))
    rr.log("info/scene", rr.TextLog("Coordinate system: First frame camera coordinates"))

    # 可视化数据
    if sample_indices is not None:
        for idx in sample_indices:
            if 0 <= idx < len(visualizer.samples):
                visualizer.visualize_sample(idx)
    else:
        visualizer.visualize_all_samples()


def batch_visualize_transformed_scenes(data_root, transformed_samples_dir, scene_ids=None, samples_per_scene=3):
    """
    批量可视化多个场景的变换采样数据

    Args:
        data_root (str): 数据根目录
        transformed_samples_dir (str): 变换采样结果目录
        scene_ids (list, optional): 要可视化的场景ID列表，None表示全部
        samples_per_scene (int): 每个场景可视化的采样数量
    """

    # 获取所有场景
    if scene_ids is None:
        scene_ids = []
        for f in os.listdir(transformed_samples_dir):
            if f.endswith('_sampling_results_transformed.json'):
                scene_id = f.replace('_sampling_results_transformed.json', '')
                scene_ids.append(scene_id)

    print(f"准备可视化 {len(scene_ids)} 个变换场景")

    for i, scene_id in enumerate(scene_ids):
        print(f"可视化变换场景 [{i + 1}/{len(scene_ids)}]: {scene_id}")

        scene_path = os.path.join(data_root, scene_id)
        transformed_json_path = os.path.join(transformed_samples_dir, f"{scene_id}_sampling_results_transformed.json")

        if not os.path.exists(scene_path) or not os.path.exists(transformed_json_path):
            print(f"跳过场景 {scene_id}，文件不存在")
            continue

        try:
            # 为每个场景创建独立的可视化
            sample_indices = list(range(min(samples_per_scene, 10)))  # 限制最多10个采样
            visualize_transformed_scene_samples(
                scene_path,
                transformed_json_path,
                sample_indices=sample_indices,
                app_name=f"transformed_scene_{scene_id}"
            )

        except Exception as e:
            print(f"可视化变换场景 {scene_id} 时出错: {str(e)}")
            continue


def main():
    """
    主函数，支持命令行参数
    """
    parser = argparse.ArgumentParser(description="可视化变换后的采样数据（First Frame Coordinates）")
    parser.add_argument("--scene_path", type=str, required=True, help="场景数据路径")
    parser.add_argument("--transformed_json", type=str, required=True, help="变换后的采样JSON文件路径")
    parser.add_argument("--sample_indices", type=int, nargs="+", default=None,
                        help="指定要可视化的采样索引列表（可选）")
    parser.add_argument("--app_name", type=str, default=None, help="rerun应用名称")

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.scene_path):
        print(f"错误: 场景路径不存在: {args.scene_path}")
        return

    if not os.path.exists(args.transformed_json):
        print(f"错误: 变换采样JSON文件不存在: {args.transformed_json}")
        return

    # 可视化数据
    visualize_transformed_scene_samples(
        args.scene_path,
        args.transformed_json,
        sample_indices=args.sample_indices,
        app_name=args.app_name
    )


if __name__ == "__main__":

    scene = "42897951"
    scene_path = f"/media/lyq/temp/dataset/train-CA-1M-slam/{scene}"
    transformed_json_path = f"/media/lyq/temp/dataset/train-CA-1M-slam/{scene}/{scene}_sampling_results_transformed.json"

    # 可视化特定采样
    idx = 53
    visualize_transformed_scene_samples(
        scene_path,
        transformed_json_path,
        sample_indices=[0],
        app_name=f"transformed{idx}_{scene}"
    )

    # 批量可视化
    # data_root = "/data/CA-1M-slam/"
    # transformed_samples_dir = "/data/CA-1M-sample-claude-v4-transformed/"
    # batch_visualize_transformed_scenes(data_root, transformed_samples_dir, samples_per_scene=3)