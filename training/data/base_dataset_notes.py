# 版权声明：Meta Platforms, Inc. 及其关联公司
# 所有权利保留
#
# 本源代码遵循根目录LICENSE文件中的许可协议

import numpy as np
from PIL import Image, ImageFile

from torch.utils.data import Dataset
from .dataset_util import *  # 导入自定义的数据集工具函数

# 允许加载超大图像文件
Image.MAX_IMAGE_PIXELS = None
# 允许加载截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    """
    VGGT和VGGSfM训练的基类数据集
    
    这个抽象类处理图像大小调整、数据增强和坐标转换等通用操作。
    具体的数据集实现应继承此类。
    
    属性:
        img_size: 目标图像尺寸（通常指宽度）
        patch_size: ViT模型的patch大小
        augs.scales: 数据增强的尺度范围[min, max]
        rescale: 是否缩放图像
        rescale_aug: 缩放时是否应用增强
        landscape_check: 是否处理横屏与竖屏方向
    """
    def __init__(
        self,
        common_conf,  # 包含所有数据集共享配置的对象
    ):
        """
        使用通用配置初始化基类数据集
        
        参数:
            common_conf: 包含以下属性的配置对象，所有数据集共享:
                - img_size: 默认518
                - patch_size: 默认14
                - augs.scales: 默认[0.8, 1.2]
                - rescale: 默认True
                - rescale_aug: 默认True
                - landscape_check: 默认True
        """
        super().__init__()
        # 初始化配置参数
        self.img_size = common_conf.img_size  # 目标图像尺寸
        self.patch_size = common_conf.patch_size  # ViT的patch大小
        self.aug_scale = common_conf.augs.scales  # 数据增强的尺度范围
        self.rescale = common_conf.rescale  # 是否缩放图像
        self.rescale_aug = common_conf.rescale_aug  # 缩放时是否应用增强
        self.landscape_check = common_conf.landscape_check  # 是否检查横竖屏方向

    def __len__(self):
        return self.len_train  # 返回训练集长度，需在子类中实现

    def __getitem__(self, idx_N):
        """
        从数据集中获取一个项目
        
        参数:
            idx_N: 包含(seq_index, img_per_seq, aspect_ratio)的元组
        
        返回:
            由get_data()返回的数据集项目
        """
        # 解包索引元组
        seq_index, img_per_seq, aspect_ratio = idx_N
        # 调用get_data方法获取数据
        return self.get_data(
            seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio
        )

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0):
        """
        抽象方法：获取指定序列的数据
        
        参数:
            seq_index (int, 可选): 序列索引
            seq_name (str, 可选): 序列名称
            ids (list, 可选): 帧ID列表
            aspect_ratio (float, 可选): 目标宽高比
        
        返回:
            数据集特定数据
        
        抛出:
            NotImplementedError: 子类必须实现此方法
        """
        # 抽象方法占位符，强制子类实现具体逻辑
        raise NotImplementedError(
            "这是一个抽象方法，应在子类中实现，即每个数据集都应实现自己的get_data方法。"
        )

    def get_target_shape(self, aspect_ratio):
        """
        根据给定宽高比计算目标形状
        
        参数:
            aspect_ratio: 目标宽高比
        
        返回:
            numpy.ndarray: 目标图像形状[高度, 宽度]
        """
        # 计算短边尺寸
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size  # ViT的patch大小

        # 确保输入形状兼容ViT（能被patch大小整除）
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size

        # 构建最终图像形状数组
        image_shape = np.array([short_size, self.img_size])
        return image_shape

    def process_one_image(
        self,
        image,  # 输入图像数组
        depth_map,  # 深度图数组
        extri_opencv,  # OpenCV格式的外参矩阵
        intri_opencv,  # OpenCV格式的内参矩阵
        original_size,  # 原始图像尺寸[高度, 宽度]
        target_image_shape,  # 目标图像形状
        track=None,  # 可选的追踪信息
        filepath=None,  # 用于调试的文件路径
        safe_bound=4,  # 裁剪操作的安全边界
    ):
        """
        处理单张图像及其关联数据
        
        处理图像变换、深度处理和坐标转换
        
        参数:
            image: 输入图像数组
            depth_map: 深度图数组
            extri_opencv: OpenCV格式的外参矩阵
            intri_opencv: OpenCV格式的内参矩阵
            original_size: 原始图像尺寸[高度, 宽度]
            target_image_shape: 处理后的目标图像形状
            track: 可选的追踪信息
            filepath: 用于调试的文件路径
            safe_bound: 裁剪操作的安全边界
        
        返回:
            tuple: 包含处理后的:
                - 图像
                - 深度图
                - 更新后的外参矩阵
                - 更新后的内参矩阵
                - 世界坐标系中的3D点
                - 相机坐标系中的3D点
                - 有效点的布尔掩码
                - 更新后的追踪信息（如果有）
        """
        # 创建副本避免原地操作影响原始数据
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)

        # 训练时应用随机尺度增强（如果启用）
        if self.training and self.aug_scale:
            # 在指定范围内生成随机缩放因子
            random_h_scale, random_w_scale = np.random.uniform(
                self.aug_scale[0], self.aug_scale[1], 2
            )
            # 通过上限1.0避免随机填充
            random_h_scale = min(random_h_scale, 1.0)
            random_w_scale = min(random_w_scale, 1.0)
            # 计算增强后的尺寸
            aug_size = original_size * np.array([random_h_scale, random_w_scale])
            aug_size = aug_size.astype(np.int32)
        else:
            aug_size = original_size  # 无增强时使用原始尺寸

        # 将主点移至图像中心并根据需要裁剪
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath,
        )

        # 更新原始尺寸
        original_size = np.array(image.shape[:2])
        target_shape = target_image_shape

        # 处理横屏与竖屏方向
        rotate_to_portrait = False
        if self.landscape_check:
            # 必要时在横竖屏间切换
            if original_size[0] > 1.25 * original_size[1]:
                if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
                    target_shape = np.array([target_image_shape[1], target_image_shape[0]])
                    rotate_to_portrait = True  # 标记需要旋转

        # 调整图像大小并更新内参
        if self.rescale:
            image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
                image, depth_map, intri_opencv, target_shape, original_size, track=track,
                safe_bound=safe_bound,
                rescale_aug=self.rescale_aug  # 应用缩放增强
            )
        else:
            print("不缩放图像")

        # 确保最终裁剪到目标形状
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath, strict=True,
        )

        # 应用90度旋转（如果需要）
        if rotate_to_portrait:
            assert self.landscape_check
            # 随机选择顺时针或逆时针
            clockwise = np.random.rand() > 0.5
            image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                clockwise=clockwise,  # 旋转方向
                track=track,  # 更新追踪信息
            )

        # 将深度转换为世界和相机坐标系中的点
        world_coords_points, cam_coords_points, point_mask = (
            depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
        )

        # 返回处理后的所有数据
        return (
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            world_coords_points,
            cam_coords_points,
            point_mask,
            track,
        )

    def get_nearby_ids(self, ids, full_seq_num, expand_ratio=None, expand_range=None):
        """
        TODO: 添加基于位姿相似度排名的采样函数
        
        从序列中采样一组接近起始索引的ID
        
        可以通过扩展比例或固定窗口大小指定范围
        
        参数:
            ids: 初始ID列表，第一个元素用作锚点
            full_seq_num: 完整序列的总项数
            expand_ratio: 围绕起始索引扩展ID数量的比例因子
                如果未提供expand_ratio和expand_range，默认为2.0
            expand_range: 围绕起始索引扩展的固定项数
                如果提供，则忽略expand_ratio
        
        返回:
            numpy.ndarray: 采样的ID数组，第一个元素是原始起始索引
        
        示例:
            # 使用expand_ratio（默认行为）
            # 如果ids=[100,101,102]且full_seq_num=200，expand_ratio=2.0，
            # expand_range = int(3 * 2.0) = 6，ID从[94...106]采样（如果边界允许）
            
            # 直接使用expand_range
            # 如果ids=[100,101,102]且full_seq_num=200，expand_range=10，
            # ID从[90...110]采样（如果边界允许）
        
        抛出:
            ValueError: 如果未提供ID
        """
        # 检查ID列表是否为空
        if len(ids) == 0:
            raise ValueError("未提供ID。")

        # 设置默认扩展比例（如果未提供任何参数）
        if expand_range is None and expand_ratio is None:
            expand_ratio = 2.0  # 默认行为

        # 获取ID数量和起始索引
        total_ids = len(ids)
        start_idx = ids[0]

        # 确定实际的扩展范围
        if expand_range is None:
            # 使用比例确定范围
            expand_range = int(total_ids * expand_ratio)

        # 计算有效边界
        low_bound = max(0, start_idx - expand_range)  # 下限（不低于0）
        high_bound = min(full_seq_num, start_idx + expand_range)  # 上限（不超过序列总数）

        # 创建有效索引范围
        valid_range = np.arange(low_bound, high_bound)

        # 采样total_ids - 1个项目（因为起始索引已包含）
        sampled_ids = np.random.choice(
            valid_range,
            size=(total_ids - 1),
            replace=True,   # 允许采样到相同ID
        )

        # 在开头插入起始索引
        result_ids = np.insert(sampled_ids, 0, start_idx)

        return result_ids