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

class CA1MDataset(BaseDataset):
    def __init__(
        self,
        common_conf, # 通用配置参数
        split: str = "train", # 数据集分割（训练集/测试集）
        CA1M_DIR: str = None, # CA1M数据集根目录路径
        CA1M_ANNOTATION_DIR: str = None, # CA1M标注文件目录路径
        min_num_images: int = 2, # 24 序列最小图像数量要求
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

            for seq_name in seq_list:
                data_json_path = osp.join(
                    self.CA1M_DIR, seq_name, f"{seq_name}_sampling_results_transformed.json"
                )
                with open(data_json_path, 'r') as f:
                    single_seq_data = json.load(f)
                
                # 跳过图像数不足的序列
                if len(single_seq_data) < min_num_images:
                    continue
                # 跳过无效序列
                if seq_name in self.invalid_sequence:
                    continue
                # 更新总帧数
                total_frame_num += len(single_seq_data)
                # 存储序列数据
                self.data_store[seq_name] = single_seq_data
        # 初始化序列列表和计数器
        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num
        # 记录数据集统计信息
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: CA1M Data size: {self.sequence_list_len}")
        logging.info(f"{status}: CA1M Data dataset length: {len(self)}")
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
        # 内部随机选择序列
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        # 获取序列名称
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]
        # 获取序列元数据
        metadata = self.data_store[seq_name]
        # 如果没有提供特定ID，则随机选择图像
        if ids is None:
            ids = np.asarray(random.randint(0, len(metadata) - 1))
        # 获取选定ID的标注
        annos = metadata[ids] # dict_keys(['image_id', 'bbox_id'])
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
        
        image_idxs = annos["image_id"]  # 获取图像ID
        seq_poses = np.load(osp.join(self.CA1M_DIR, seq_name, 'all_poses.npy')).reshape(-1, 4, 4)  # 获取序列位姿
        K_rgb = np.loadtxt(osp.join(self.CA1M_DIR, seq_name, 'K_rgb.txt')).reshape(3,3)
        
        # print("Seq", seq_name,'ids', ids)
        # Load Box GT information
        bbox_corners = np.asarray(annos["corners"])  # 获取边界框ID

        bbox_corners = self.process_bbox_corners(bbox_corners)  # 处理边界框角点
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
                depth_map = read_depth(depth_path, 1000.0) # 1.0
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
            # 获取外参和内参矩阵
            extri_opencv = seq_poses[img_idx][:3,:] #np.array(anno["extri"])
            intri_opencv = K_rgb #np.array(anno["intri"])
            # 处理单张图像
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
            # 添加处理后的数据到列表
            images.append(image) # np
            depths.append(depth_map)
            extrinsics.append(extri_opencv) # [3,4]
            intrinsics.append(intri_opencv) # [3,3]
            cam_points.append(cam_coords_points) # [H,W,3]
            world_points.append(world_coords_points) # [H,W,3]
            point_masks.append(point_mask) # [H,W]
            image_paths.append(image_path) # [H,W]
            original_sizes.append(original_size)
        # print("len images", len(images))
        # 数据集名称
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
    
if __name__ == "__main__":
    test_json = '/media/lyq/temp/dataset/train-CA-1M-slam/47204724/47204724_sampling_results.json'
    with open(test_json, 'r') as f:
        seq_data = json.load(f)
    print("len seq_data:", len(seq_data))