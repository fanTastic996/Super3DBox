# 版权声明
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的库
import gzip
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np

# 从自定义模块导入工具函数和基础数据集类
from data.dataset_util import *
from data.base_dataset import BaseDataset

# 定义Co3D数据集中包含的类别列表
SEEN_CATEGORIES = [
    "apple", "backpack", "banana", "baseballbat", "baseballglove", "bench", "bicycle", 
    "bottle", "bowl", "broccoli", "cake", "car", "carrot", "cellphone", "chair", "cup", 
    "donut", "hairdryer", "handbag", "hydrant", "keyboard", "laptop", "microwave", 
    "motorcycle", "mouse", "orange", "parkingmeter", "pizza", "plant", "stopsign", 
    "teddybear", "toaster", "toilet", "toybus", "toyplane", "toytrain", "toytruck", 
    "tv", "umbrella", "vase", "wineglass",
]

class Co3dDataset(BaseDataset):
    def __init__(
        self,
        common_conf,       # 通用配置参数
        split: str = "train",  # 数据集分割（训练集/测试集）
        CO3D_DIR: str = None,  # CO3D数据集根目录路径
        CO3D_ANNOTATION_DIR: str = None,  # CO3D标注文件目录路径
        min_num_images: int = 24,  # 序列最小图像数量要求
        len_train: int = 100000,   # 训练集长度
        len_test: int = 10000,     # 测试集长度
    ):
        """
        初始化Co3D数据集

        参数:
            common_conf: 包含通用配置的对象
            split: 数据集分割（'train'或'test'）
            CO3D_DIR: CO3D数据集根目录路径
            CO3D_ANNOTATION_DIR: CO3D标注文件目录路径
            min_num_images: 序列的最小图像数量要求
            len_train: 训练集长度
            len_test: 测试集长度

        异常:
            ValueError: 如果未指定CO3D_DIR或CO3D_ANNOTATION_DIR
        """
        # 调用父类初始化方法
        super().__init__(common_conf=common_conf)

        # 从配置中获取参数
        self.debug = common_conf.debug  # 调试模式
        self.training = common_conf.training  # 训练模式标志
        self.get_nearby = common_conf.get_nearby  # 是否获取邻近帧
        self.load_depth = common_conf.load_depth  # 是否加载深度图
        self.inside_random = common_conf.inside_random  # 内部随机选择序列
        self.allow_duplicate_img = common_conf.allow_duplicate_img  # 允许重复图像

        # 检查必要的目录路径是否提供
        if CO3D_DIR is None or CO3D_ANNOTATION_DIR is None:
            raise ValueError("必须指定CO3D_DIR和CO3D_ANNOTATION_DIR路径")

        # 使用预定义的类别列表
        category = sorted(SEEN_CATEGORIES)

        # 调试模式下只使用'tv'类别
        if self.debug: 
            category = ['tv']

        # 根据数据集分割设置
        if split == "train":
            split_name_list = ["train"]  # 训练集分割
            self.len_train = len_train   # 训练集长度
        elif split == "test":
            split_name_list = ["test"]   # 测试集分割
            self.len_train = len_test     # 测试集长度
        else:
            raise ValueError(f"无效的分割类型: {split}")

        # 初始化数据集存储结构
        self.invalid_sequence = []  # 无效序列列表
        self.category_map = {}      # 类别映射
        self.data_store = {}        # 数据存储字典
        self.seqlen = None          # 序列长度
        self.min_num_images = min_num_images  # 序列最小图像数

        # 记录数据集目录信息
        logging.info(f"CO3D数据集目录: {CO3D_DIR}")

        # 存储路径
        self.CO3D_DIR = CO3D_DIR
        self.CO3D_ANNOTATION_DIR = CO3D_ANNOTATION_DIR

        total_frame_num = 0  # 总帧数计数器

        # 遍历所有类别和分割
        for c in category:
            for split_name in split_name_list:
                # 构建标注文件路径
                annotation_file = osp.join(
                    self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz"
                )

                try:
                    # 读取并解析gzip压缩的标注文件
                    with gzip.open(annotation_file, "r") as fin:
                        annotation = json.loads(fin.read())
                        
                except FileNotFoundError:
                    # 处理文件不存在的情况
                    logging.error(f"找不到标注文件: {annotation_file}")
                    continue

                # 处理每个序列的数据
                for seq_name, seq_data in annotation.items():
                    # 跳过图像数不足的序列
                    if len(seq_data) < min_num_images:
                        continue
                    # 跳过无效序列
                    if seq_name in self.invalid_sequence:
                        continue
                    # 更新总帧数
                    total_frame_num += len(seq_data)
                    # 存储序列数据
                    self.data_store[seq_name] = seq_data

        # 初始化序列列表和计数器
        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        # 记录数据集统计信息
        status = "训练" if self.training else "测试"
        logging.info(f"{status}: Co3D 序列数量: {self.sequence_list_len}")
        logging.info(f"{status}: Co3D 数据集长度: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,  # 序列索引
        img_per_seq: int = None,  # 每序列图像数
        seq_name: str = None,    # 序列名称
        ids: list = None,        # 特定图像ID列表
        aspect_ratio: float = 1.0,  # 图像宽高比
    ) -> dict:
        """
        获取指定序列的数据

        参数:
            seq_index: 序列索引
            img_per_seq: 每序列图像数
            seq_name: 序列名称
            ids: 特定图像ID列表
            aspect_ratio: 图像宽高比

        返回:
            包含图像、深度图等数据的字典
        """
        # 内部随机选择序列
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
            
        # 获取序列名称
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        # 获取序列元数据
        metadata = self.data_store[seq_name] # Dict
        
        # 如果没有提供特定ID，则随机选择图像
        if ids is None:
            ids = np.random.choice(
                len(metadata), img_per_seq, replace=self.allow_duplicate_img
            )

        # 获取选定ID的标注
        annos = [metadata[i] for i in ids]

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

        # 处理每个选定的图像
        for anno in annos:
            # 获取图像路径
            filepath = anno["filepath"]
            image_path = osp.join(self.CO3D_DIR, filepath)
            
            # 读取图像
            image = read_image_cv2(image_path)

            # 如果需要加载深度图
            if self.load_depth:
                # 构建深度图路径
                depth_path = image_path.replace("/images", "/depths") + ".geometric.png"
                # 读取深度图
                depth_map = read_depth(depth_path, 1.0)

                # 构建掩码路径
                mvs_mask_path = image_path.replace(
                    "/images", "/depth_masks"
                ).replace(".jpg", ".png")
                # 读取并处理掩码
                mvs_mask = cv2.imread(mvs_mask_path, cv2.IMREAD_GRAYSCALE) > 128
                depth_map[~mvs_mask] = 0

                # 阈值处理深度图
                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=98
                )
            else:
                depth_map = None  # 不加载深度图

            # 获取原始尺寸
            original_size = np.array(image.shape[:2])
            # 获取外参和内参矩阵
            extri_opencv = np.array(anno["extri"])
            intri_opencv = np.array(anno["intri"])

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
            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        # 数据集名称
        set_name = "co3d"

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
            "point_masks": point_masks,      # 点掩码列表
            "original_sizes": original_sizes,  # 原始尺寸列表
        }
        return batch