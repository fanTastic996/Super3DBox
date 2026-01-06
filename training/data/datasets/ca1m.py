# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np


from data.dataset_util import *
from data.base_dataset import BaseDataset
import torch
import open3d as o3d
from scipy.spatial import KDTree
from vggt.utils.rotation import mat_to_quat, quat_to_mat

class CA1MDataset(BaseDataset):
    def __init__(
        self,
        common_conf, # 通用配置参数
        split: str = "train", # 数据集分割（训练集/测试集）
        CA1M_DIR: str = None, # CA1M数据集根目录路径
        CA1M_ANNOTATION_DIR: str = None, # CA1M标注文件目录路径
        min_num_images: int = 50, # 24 序列最小图像数量要求
        len_train: int = 100000, # 训练集长度
        len_test: int = 10000, # 测试集长度
    ):
        """
        Initialize the CA1MDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            CA1M_DIR (str): Directory path to CA1M data.
            CA1M_ANNOTATION_DIR (str): Directory path to CA1M annotations.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If CA1M_DIR or CA1M_ANNOTATION_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug # 调试模式
        self.training = common_conf.training # 训练模式标志
        self.get_nearby = common_conf.get_nearby # 是否获取邻近帧
        self.load_depth = common_conf.load_depth # 是否加载深度图
        self.inside_random = common_conf.inside_random  # 内部随机选择序列
        self.allow_duplicate_img = common_conf.allow_duplicate_img # 允许重复图像

        if CA1M_DIR is None or CA1M_ANNOTATION_DIR is None:
            raise ValueError("Both CA1M_DIR and CA1M_ANNOTATION_DIR must be specified.")

 
        # 根据数据集分割设置
        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train # 训练集长度
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test # 测试集长度
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = [] # set any invalid sequence names here
        # 初始化数据集存储结构
        self.category_map = {}  # 类别映射
        self.data_store = {} # 数据存储字典
        self.sequence_list = []
        self.seqlen = None  
        self.min_num_images = min_num_images # 序列最小图像数
        # 记录数据集目录信息
        logging.info(f"CA1M_DIR is {CA1M_DIR}")
        # 存储路径
        self.CA1M_DIR = CA1M_DIR
        self.CA1M_ANNOTATION_DIR = CA1M_ANNOTATION_DIR

        total_frame_num = 0
        # 遍历所有序列，拿到每个序列的图像数据信息，存到self.data_store中
        for split_name in split_name_list: #["test", "train"]
            # 构建标注文件路径
            annotation_file = osp.join(
                self.CA1M_ANNOTATION_DIR, f"{split_name}_list.txt"
            )
            try:
                seq_list = np.loadtxt(annotation_file, dtype=str)
            except FileNotFoundError:
                logging.error(f"Annotation file not found: {annotation_file}")
                continue
            # 处理每个序列的数据
            print(annotation_file, "seq_list length", len(seq_list))
            for seq_name in seq_list:
                
                seq_dir = osp.join(self.CA1M_DIR, seq_name)
                seq_rgb_dir = osp.join(seq_dir, 'rgb')
                len_seq_imgs = len(os.listdir(seq_rgb_dir)) # 获取序列图像数量
                # 跳过图像数不足的序列
                if len_seq_imgs < min_num_images:
                    continue
                # 跳过无效序列
                if seq_name in self.invalid_sequence:
                    continue
                # 更新总帧数
                total_frame_num += len_seq_imgs
                self.data_store[seq_name] = len_seq_imgs
                self.sequence_list.append(seq_name)
        # 初始化序列列表和计数器
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num
        # 记录数据集统计信息
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: CA1M Data size: {self.sequence_list_len}")
        logging.info(f"{status}: CA1M Data dataset length: {self.total_frame_num}")
        
    def get_data(
        self,
        seq_index: int = None, # 序列索引
        img_per_seq: int = None,  # 每序列图像数
        seq_name: str = None, # 序列名称
        ids: list = None, # 特定图像ID列表
        aspect_ratio: float = 1.0, # 图像宽高比
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        # while True:
        # 内部随机选择序列
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        # 获取序列名称
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]
        # 指定seq
        # seq_name = '42444750'
        # seq_name = '47334115'
        # 如果没有提供特定ID，则随机选择图像
        if ids is None:
            # ids = np.random.choice(
            #     self.data_store[seq_name], img_per_seq, replace=self.allow_duplicate_img
            # )
            # ids = np.array([4, 14, 24])
            # ids = np.array([225, 240])
            # ids = np.array([460, 520])
            # ids = np.array([520])
            
            # RANDOM SAMPLE
            interval = 10
            max_start = self.data_store[seq_name] - (img_per_seq - 1) * interval
            # 随机选择一个起始索引（在安全范围内）
            start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
            # 生成等间隔的索引数组
            ids = np.array([start_idx + i * interval for i in range(img_per_seq)])
            # np.random.shuffle(ids) #打乱顺序
            
        image_idxs = ids  # 获取图像ID
        #TODO:
        # seq_name = '42444750'
        
        
        # print("Seq", seq_name,'ids', ids) 
        self.seqname= seq_name
        # Load sequence data
        scene_data = self.load_scene_data(seq_name)

        
        # 获取目标图像尺寸
        target_image_shape = self.get_target_shape(aspect_ratio)
        # 初始化数据列表
        images = []          # 图像列表
        depths = []          # 深度图列表
        cam_points = []      # 相机坐标系点列表
        world_points = []    # 世界坐标系点列表
        point_masks = []     # 点掩码列表
        extrinsics = []      # 外参矩阵列表
        intrinsics = []      # 内参矩阵列表
        image_paths = []     # 图像路径列表
        original_sizes = []  # 原始尺寸列表
        all_gravity = []
        
        # TODO:debug only train 12 th box
        # filtered_bbox_corners = filtered_bbox_corners[2:3,...] #2 is the chair
        
        
        
        seq_poses = scene_data['poses'].reshape(-1, 4, 4)  # 获 取序列位姿
        #TODO: seems right, not sure 25-8-28
        seq_poses = closed_form_inverse_se3(seq_poses).reshape(-1, 4, 4)
        
        json_directory = osp.join(self.CA1M_DIR, seq_name, 'instances')
        K_rgb = scene_data['K'] #scene_data['K_rgb']
        seq_gravity = scene_data['gravity']
         
        for img_idx in image_idxs:
            
            filepath = osp.join(seq_name,'rgb',str(img_idx)+'.png') 
            # 获取图像路径
            image_path = osp.join(self.CA1M_DIR, seq_name,'rgb',str(img_idx)+'.png') 
            # 读取图像
            image = read_image_cv2(image_path)
            # 如果需要加载深度图
            
            # self.load_depth = False # set by lyq
            if self.load_depth: #True
                depth_path = image_path.replace("/rgb", "/depth")
                # 读取深度图
                depth_map = read_depth(depth_path, 0.001) # 1.0
                # 构建掩码路径
                # mvs_mask_path = image_path.replace(
                #     "/images", "/depth_masks"
                # ).replace(".jpg", ".png")
                # # 读取并处理掩码
                # mvs_mask = cv2.imread(mvs_mask_path, cv2.IMREAD_GRAYSCALE) > 128
                mvs_mask = np.ones_like(depth_map, dtype=bool) # 全部为True
                depth_map[~mvs_mask] = 0
                # 阈值处理深度图
                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=98
                )
            else:
                depth_map = None # 不加载深度图
            # 获取原始尺寸
            original_size = np.array(image.shape[:2])
            # rescale RGB image to the same size of depth map
            if original_size[0] != depth_map.shape[0] or original_size[1] != depth_map.shape[1]:
                image = cv2.resize(image, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_AREA)
                original_size = np.array(image.shape[:2])
            
            
            # 获取外参和内参矩阵
            cur_pose = seq_poses[img_idx] # [4,4]
            extri_opencv = cur_pose[:3,:] # np.array(anno["extri"])
            intri_opencv = K_rgb # np.array(anno["intri"])
            
            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=filepath,
            )
            # print(filepath,"image", image.shape)
            # print(filepath,"depth_map",depth_map.shape)
            # print(filepath,"extri_opencv",extri_opencv.shape)
            # print(filepath,"intri_opencv",intri_opencv)
            

            
            # 添加处理后的数据到列表
            images.append(image) 
            depths.append(depth_map)
            extrinsics.append(extri_opencv) # [3,4]
            intrinsics.append(intri_opencv) # [3,3]
            cam_points.append(cam_coords_points) # [H,W,3]
            world_points.append(world_coords_points) # [H,W,3]
            point_masks.append(point_mask) # [H,W]
            image_paths.append(image_path) # [H,W]
            original_sizes.append(original_size)
            all_gravity.append(seq_gravity[img_idx])

        
        '''
        GT box extraction according to the selected images
        ''' 
        # Load Box GT information
        # filtered_bbox_corners = self.filter_gt_boxes_for_images(scene_data, image_idxs, np.stack(intrinsics, axis=0), np.stack(depths, axis=0), frustum_threshold=4) # 4
        
        
        extrinsics_tmp = np.stack(extrinsics, axis=0)  # [N,3,4]
        N = extrinsics_tmp.shape[0]
        # 拼接一组[0,0,0,1]到每个外参，得到[N,4,4]
        bottom = np.tile(np.array([0, 0, 0, 1], dtype=extrinsics_tmp.dtype), (N,1)).reshape(N,1,4)
        extrinsics_tmp = np.concatenate([extrinsics_tmp, bottom], axis=1)  # [N,4,4]
        
        filtered_bbox_corners = merge_scene_gt_corners_world_multiframe(
            data_path=json_directory,
            image_idx=image_idxs,
            extrinsic=extrinsics_tmp,
            json_name_fmt="{idx}.json",
            extrinsic_is_w2c=True,   # set False if you pass in c2w
            keep_single_view=True,
            device="cpu",
        )
        
        # filter boxes according to the box  visibility
        filtered_bbox_corners, _, _, ratio = filter_gt_boxes_by_2d_valid_area_ratio_np(filtered_bbox_corners, np.stack(intrinsics, axis=0), extrinsics_tmp, H=images[0].shape[0], W=images[0].shape[1], thr=0.2, extrinsic_is_c2w=False, return_debug=True)
        
        
        if isinstance(filtered_bbox_corners, np.ndarray) and filtered_bbox_corners.size == 0:
            print(f"No valid GT boxes found for seq {seq_name} with image ids {ids}. using fake GT...")
            filtered_bbox_corners = np.zeros((1, 8, 3), dtype=np.float32)  # 使用空的GT boxes
        # change boxes that are not parallel to Z-AXIS to be parallel
        # filtered_bbox_corners, tilted_mask, tilt_degs = self.upright_boxes(filtered_bbox_corners, tilt_deg_thresh=5.0)
        # padding invalid boxes
        bbox_corners = self.process_bbox_corners(filtered_bbox_corners)  # 处理边界框角点



        
        set_name = "CA1M"
        # 构建并返回批次数据
        batch = {
            "seq_name": set_name + "_" + seq_name,  # 序列名称
            "ids": ids,                     # 图像ID列表
            "frame_num": len(extrinsics),   # 帧数
            "images": images,                # 图像列表
            "depths": depths,                # 深度图列表
            "extrinsics": extrinsics,        # 外参矩阵列表
            "intrinsics": intrinsics,        # 内参矩阵列表
            "cam_points": cam_points,        # 相机坐标系点列表
            "world_points": world_points,    # 世界坐标系点列表
            "bbox_corners": bbox_corners, # 边界框角点列表
            "point_masks": point_masks,      # 点掩码列表
            "original_sizes": original_sizes,  # 原始尺寸列表
            "gravity": all_gravity
        }
        return batch


    def process_bbox_corners(self, bbox_corners: np.ndarray) -> np.ndarray:
        """
        处理bbox_corners数组：
        - 若第一维数量不足500，填充0值补足500
        - 若第一维数量超过500，截取前500个
        - 若正好500个，直接返回
        
        参数:
            bbox_corners: 输入的三维数组，形状应为 (N, 8, 3)
            
        返回:
            处理后的三维数组，形状为 (500, 8, 3)
        """
        current_size = bbox_corners.shape[0]
        
        # 情况1：不足500时进行填充
        if current_size < 1000:
            # 创建目标数组并填充原始数据
            padded = np.zeros((1000, 8, 3), dtype=bbox_corners.dtype)
            # print("bbox_corners shape:", bbox_corners.shape)
            # print("current_size:", current_size)
            # print("pad_size:", padded.shape)
            padded[:current_size] = bbox_corners
            return padded
        
        # 情况2：超过500时截断
        elif current_size > 1000:
            return bbox_corners[:1000]
        
        # 情况3：正好500时直接返回
        else:
            return bbox_corners
    
    
    def load_scene_data(self, scene_id):
        """加载场景数据"""
        scene_path = os.path.join(self.CA1M_DIR, scene_id)

        # 加载GT instances
        instances_path = os.path.join(scene_path, 'instances.json')
        with open(instances_path, 'r') as f:
            instances_data = json.load(f)

        # 验证并提取有效的corners数据
        corners_array, valid_instances = self.validate_and_extract_corners(instances_data)
        
        gravity = np.load(os.path.join(scene_path, 'T_gravity.npy'))

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
            'scene_path': scene_path,
            'gravity': gravity
        }
    
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
    
    def filter_gt_boxes_for_images(self, scene_data, image_indices, K, depth_maps, frustum_threshold=4, dist_threshold=0.2):
        """为指定图像过滤GT boxes"""
        corners_array = scene_data['corners_array']
        # K = scene_data['K']

        # try:
        poses = scene_data['poses'][image_indices]
        # depth_maps = self.load_depth_maps(
        #     scene_data['depth_path'],
        #     image_indices,
        #     scale=0.001
        # )

        # 使用filter_gt_boxes.py的逻辑
        filtered_corners = self.filter_3d_corners(
            corners_array, K, poses, depth_maps,
            frustum_threshold=frustum_threshold, dist_threshold=dist_threshold #TODO:0.5 0.2
        )

        return filtered_corners

       
    
    def load_depth_maps(self, depth_path, image_indices, scale=0.001):
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
    
    def filter_3d_corners(self, corners, K, poses, depth_maps,
                          near=0.1, far=100.0, frustum_threshold=4, dist_threshold=0.5):  # 放宽距离阈值
        """基于filter_gt_boxes.py的过滤逻辑"""
        try:
            # 第一步：视锥体剔除
            bbox_frustum_mask = self.frustum_culling_bbox_level(
                corners, K, poses, depth_maps, near, far,
                threshold=frustum_threshold
            )   
            
            visible_bboxes = corners[bbox_frustum_mask]

            if len(visible_bboxes) == 0:
                # 如果没有可见的boxes，返回空结果
                return np.empty((0, 8, 3)), np.zeros(len(corners), dtype=bool)

            # 第二步：点云邻近性验证
            #TODO:根据场景整体点云，裁剪出相机可视部分的点云，否则会错误保留被遮挡的那些box,CA1M数据集可以直接使用depth，因为它的depth是根据Mesh渲染的
            gt_points = self.depth_to_pointcloud(depth_maps, K, poses)

            bbox_proximity_mask = self.check_bbox_proximity(
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
    
    
    def _normalize(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _rotation_align_a_to_b(self, a, b):
        """
        返回将向量 a 旋到向量 b 的 3x3 旋转矩阵（右手系）。
        a,b 为长度为3的一维向量。
        """
        a = self._normalize(a)
        b = self._normalize(b)
        c = np.dot(a, b)            # cos(theta)
        if c >= 1.0 - 1e-12:        # 已对齐
            return np.eye(3)
        if c <= -1.0 + 1e-12:       # 反向（180°），任选一条与 a 正交的轴
            # 找到任意与 a 不平行的基向量
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(a, tmp)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            axis = self._normalize(np.cross(a, tmp))
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

    def upright_boxes(self, corners: np.ndarray, tilt_deg_thresh: float = 5.0):
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
            cosang = np.clip(np.abs(np.dot(self._normalize(local_z), z_hat)), 0.0, 1.0)
            ang = np.degrees(np.arccos(cosang))
            tilt_degs[i] = float(ang)

            if ang > tilt_deg_thresh:
                # 旋转到 z 轴对齐（绕质心）
                R = self._rotation_align_a_to_b(local_z, z_hat)
                X_rot = (R @ X.T).T                # [8,3]
                pts_upright = X_rot + c
                upright[i] = pts_upright
                tilted_mask[i] = True
            else:
                # 已足够垂直，直接保留
                upright[i] = pts

        return upright, tilted_mask, tilt_degs
    
    def depth_to_pointcloud(self, depth_maps, K, T_wc):
        N_imgs = depth_maps.shape[0]
        all_pts = []
        for i in range(N_imgs):
            depth = depth_maps[i]
            K_cur = K[i]
            fx, fy = K_cur[0, 0], K_cur[1, 1]
            cx, cy = K_cur[0, 2], K_cur[1, 2]
            height, width = depth.shape
            
            # 生成网格坐标
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            z = depth.flatten()
            x = (u.flatten() - cx) * z / fx
            y = (v.flatten() - cy) * z / fy
            
            # 转换到世界坐标系
            points_cam = np.vstack([x, y, z]).T
            points_world = self.apply_transform(points_cam, T_wc[i])
            
            # 过滤无效点并提取颜色
            valid = z > 0
            all_pts.append(points_world[valid])
            
        all_pts = np.concatenate(all_pts, axis=0)
        
        return all_pts
    
    def apply_transform(self, points, T):
        homo_points = np.hstack([points, np.ones((len(points), 1))])
        return (T @ homo_points.T).T[:, :3]
    
    
    def frustum_culling_bbox_level(self, corners, K, poses, depth_maps, near, far, threshold=2):
        """视锥体剔除"""
        N = corners.shape[0]
        M = len(poses)
        bbox_mask = np.zeros((N, 8), dtype=bool)



        for i in range(M):
            try:
                K_cur = K[i]
                fx, fy = K_cur[0, 0], K_cur[1, 1]
                cx, cy = K_cur[0, 2], K_cur[1, 2]
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
        # print(self.seqname,"邻近性检查通过", proximity_mask.sum(), '/', len(proximity_mask))
        return proximity_mask
    
if __name__ == "__main__":
    test_json = '/media/lyq/temp/dataset/train-CA-1M-slam/47204724/47204724_sampling_results.json'
    with open(test_json, 'r') as f:
        seq_data = json.load(f)
    print("len seq_data:", len(seq_data))