"""
Dataset to stream RGB-D data from the NeRFCapture iOS App -> Cubify Transformer

Adapted from SplaTaM: https://github.com/spla-tam/SplaTAM
"""

import numpy as np
import time
import torch
import glob
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import re
import yaml
import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

from dataclasses import dataclass
from cyclonedds.domain import DomainParticipant, Domain
from cyclonedds.core import Qos, Policy
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration

from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import IterableDataset

from cubifyanything.boxes import DepthInstance3DBoxes
from cubifyanything.measurement import ImageMeasurementInfo, DepthMeasurementInfo
from cubifyanything.orientation import ImageOrientation, rotate_tensor, ROT_Z
from cubifyanything.sensor import SensorArrayInfo, SensorInfo, PosedSensorInfo
import json

#ROS
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException
import cv_bridge
import numpy as np
import queue
import threading
import time
from scipy.spatial.transform import Rotation
import cv2


def parse_transform_3x3_np(data):
    return torch.tensor(data.reshape(3, 3).astype(np.float32))

def parse_transform_4x4_np(data):
    return torch.tensor(data.reshape(4, 4).astype(np.float32))

def parse_size(data):
    return tuple(int(x) for x in data.decode("utf-8").strip("[]").split(", "))

# DDS
# ==================================================================================================
@dataclass
@annotate.final
@annotate.autoid("sequential")
class CaptureFrame(idl.IdlStruct, typename="CaptureData.CaptureFrame"):
    id: types.uint32
    annotate.key("id")
    timestamp: types.float64
    fl_x: types.float32
    fl_y: types.float32
    cx: types.float32
    cy: types.float32
    transform_matrix: types.array[types.float32, 16]
    width: types.uint32
    height: types.uint32
    image: types.sequence[types.uint8]
    has_depth: bool
    depth_width: types.uint32
    depth_height: types.uint32
    depth_scale: types.float32
    depth_image: types.sequence[types.uint8]

# 8 MB seems to work for me, but not 10 MB.
dds_config = """<?xml version="1.0" encoding="UTF-8" ?> \
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd"> \
    <Domain id="any"> \
        <Internal> \
            <MinimumSocketReceiveBufferSize>8MB</MinimumSocketReceiveBufferSize> \
        </Internal> \
        <Tracing> \
            <Verbosity>config</Verbosity> \
            <OutputFile>stdout</OutputFile> \
        </Tracing> \
    </Domain> \
</CycloneDDS> \
"""

T_RW_to_VW = np.array([[0, 0, -1, 0],
                       [-1,  0, 0, 0],
                       [0, 1, 0, 0],
                       [ 0, 0, 0, 1]]).reshape((4,4)).astype(np.float32)

T_RC_to_VC = np.array([[1,  0,  0, 0],
                       [0, -1,  0, 0],
                       [0,  0, -1, 0],
                       [0,  0,  0, 1]]).reshape((4,4)).astype(np.float32)

T_VC_to_RC = np.array([[1,  0,  0, 0],
                       [0, -1,  0, 0],
                       [0,  0, -1, 0],
                       [0,  0,  0, 1]]).reshape((4,4)).astype(np.float32)

first_frame_pose = None

def compute_VC2VW_from_RC2RW(T_RC_to_RW):
    T_vc2rw = np.matmul(T_RC_to_RW,T_VC_to_RC)
    T_vc2vw = np.matmul(T_RW_to_VW,T_vc2rw)
    return T_vc2vw

def get_camera_to_gravity_transform(pose, current, target=ImageOrientation.UPRIGHT):
    z_rot_4x4 = torch.eye(4).float()
    z_rot_4x4[:3, :3] = ROT_Z[(current, target)]
    pose = pose @ torch.linalg.inv(z_rot_4x4.to(pose))

    # This is somewhat lazy.
    fake_corners = DepthInstance3DBoxes(
        np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])).corners[:, [1, 5, 4, 0, 2, 6, 7, 3]]
    fake_corners = torch.cat((fake_corners, torch.ones_like(fake_corners[..., :1])), dim=-1).to(pose)

    fake_corners = (torch.linalg.inv(pose) @ fake_corners.permute(0, 2, 1)).permute(0, 2, 1)[..., :3]
    fake_basis = torch.stack([
        (fake_corners[:, 1] - fake_corners[:, 0]) / torch.linalg.norm(fake_corners[:, 1] - fake_corners[:, 0], dim=-1)[:, None],
        (fake_corners[:, 3] - fake_corners[:, 0]) / torch.linalg.norm(fake_corners[:, 3] - fake_corners[:, 0], dim=-1)[:, None],
        (fake_corners[:, 4] - fake_corners[:, 0]) / torch.linalg.norm(fake_corners[:, 4] - fake_corners[:, 0], dim=-1)[:, None],
    ], dim=1).permute(0, 2, 1)

    # this gets applied _after_ predictions to put it in camera space.
    T = Rotation.from_euler("xz", Rotation.from_matrix(fake_basis[-1].cpu().numpy()).as_euler("yxz")[1:]).as_matrix()

    return torch.tensor(T).to(pose)

MAX_LONG_SIDE = 1024





class MultiSensorFusion(Node):
    def __init__(self):
        super().__init__('multi_sensor_fusion_node')
        
        # 1. 性能监控
        self.frame_count = 0
        self.last_log_time = time.time()
        
        # 2. 初始化CV桥接器
        self.bridge = cv_bridge.CvBridge()
        
        # # 3. 增大队列深度
        # self.rgb_queue = queue.Queue(maxsize=50)
        # self.depth_queue = queue.Queue(maxsize=50)
        # self.pose_queue = queue.Queue(maxsize=100)
        # self.result_queue = queue.Queue(maxsize=100)  # 新增结果队列

        # 3. 增大队列深度
        self.rgb_queue = queue.Queue(maxsize=1000)
        self.depth_queue = queue.Queue(maxsize=1000)
        self.pose_queue = queue.Queue(maxsize=1000)
        self.result_queue = queue.Queue(maxsize=1000)  # 新增结果队列
        
        self.last_rgb_put_time = 0  # 上次RGB入队时间
        self.last_depth_put_time = 0  # 上次深度入队时间
        self.last_pose_put_time = 0  # 上次深度入队时间
        self.last_pose_put_time = 0  # 上次深度入队时间
        self.MIN_INTERVAL = 0.0  # 50ms最小间隔

        # 4. TF监听优化
        self.tf_buffer = Buffer(cache_time=rclpy.time.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.source_frame = 'map'
        self.target_frame = 'camera_link'
        
        #5. 图像订阅优化（降低QoS深度）
        self.rgb_sub = self.create_subscription(
            Image, '/rosefusion/rgb', self.rgb_callback, 5)  # QoS=5
        
        self.depth_sub = self.create_subscription(
            Image, '/rosefusion/depth', self.depth_callback, 5)
        
        # self.rgb_sub = self.create_subscription(
        #     Image, '/rgb/image_raw', self.rgb_callback, 5)  # QoS=5
        
        # self.depth_sub = self.create_subscription(
        #     Image, '/depth/image_raw', self.depth_callback, 5)


        # 6. 位姿获取改为定时器驱动
        self.pose_timer = self.create_timer(0.01, self.pose_update)  # 50Hz
        
        # 7. 同步处理优化
        self.sync_timer = self.create_timer(0.01, self.process_synced_data)  # 30Hz
        self.data_callback = None  # 新增回调函数属性

        self.get_logger().info("🚀 多传感器融合节点（优化版）已启动")

    def set_data_callback(self, callback):
        """设置外部数据回调函数"""
        self.data_callback = callback

    def rgb_callback(self, msg):
        """优化后的RGB回调"""
        current_time = time.monotonic()
        if current_time - self.last_rgb_put_time < self.MIN_INTERVAL:
            return  # 跳过帧
        try:
            # 移除缩放操作
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            timestamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            self.rgb_queue.put((timestamp, cv_image), timeout=0.001)
            self.last_rgb_put_time = current_time  # 更新最后入队时间
        except Exception as e:
            self.get_logger().warn(f"RGB入队异常: {str(e)}")

    def depth_callback(self, msg):
        """优化后的深度回调"""
        current_time = time.monotonic()
        if current_time - self.last_depth_put_time < self.MIN_INTERVAL:
            return  # 跳过帧
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            timestamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            self.depth_queue.put((timestamp, depth_image), timeout=0.001)
            self.last_depth_put_time = current_time
        except Exception as e:
            self.get_logger().warn(f"深度入队异常: {str(e)}")

    def pose_update(self):
        """定时获取位姿（50Hz）"""
        current_time = time.monotonic()
        if current_time - self.last_pose_put_time < self.MIN_INTERVAL:
            return  # 跳过帧
        try:
            if self.tf_buffer.can_transform(
                self.source_frame, 
                self.target_frame, 
                rclpy.time.Time()
            ):
                transform = self.tf_buffer.lookup_transform(
                    self.source_frame,
                    self.target_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.time.Duration(seconds=0.05)
                )
                
                # 转换为4x4矩阵
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                pose_matrix = self._quaternion_to_matrix(
                    translation.x, translation.y, translation.z,
                    rotation.x, rotation.y, rotation.z, rotation.w
                )
                
                # 获取时间戳
                stamp = transform.header.stamp
                timestamp = stamp.sec * 10**9 + stamp.nanosec
                
                # 放入队列
                self.pose_queue.put((timestamp, pose_matrix), timeout=0.001)
                self.last_pose_put_time = current_time
        except (TransformException, queue.Full) as e:
            pass
    
    def _quaternion_to_matrix(self, x, y, z, qx, qy, qz, qw):
        """将四元数转换为4x4位姿矩阵"""
        rot = Rotation.from_quat([qx, qy, qz, qw])
        rotation_matrix = rot.as_matrix()
        
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[0, 3] = x
        pose_matrix[1, 3] = y
        pose_matrix[2, 3] = z
        return pose_matrix


    def process_synced_data(self):
        """智能三路数据同步处理（支持数据丢弃和重匹配）"""
        try:
            MAX_RGB_DEPTH_DIFF = 0 * 1e6  # 30ms = 30,000,000 ns
            MAX_POSE_DIFF = 0 * 1e6  # 50ms = 50,000,000 ns
            
            # 1. 收集所有可用数据
            rgb_candidates = []
            depth_candidates = []
            pose_candidates = []
            
            # 获取RGB数据（最多收集3帧）
            for _ in range(100):
                try:
                    rgb_item = self.rgb_queue.get(timeout=0.001)
                    # 验证数据格式
                    if isinstance(rgb_item, tuple) and len(rgb_item) == 2:
                        rgb_candidates.append(rgb_item)
                    else:
                        self.get_logger().warn(f"RGB数据格式错误: {type(rgb_item)}, 长度: {len(rgb_item) if hasattr(rgb_item, '__len__') else 'N/A'}")
                except queue.Empty:
                    break
                except Exception as e:
                    self.get_logger().warn(f"RGB数据获取异常: {str(e)}")
                    break
            
            # 获取深度数据（最多收集3帧）
            for _ in range(100):
                try:
                    depth_item = self.depth_queue.get(timeout=0.001)
                    # 验证数据格式
                    if isinstance(depth_item, tuple) and len(depth_item) == 2:
                        depth_candidates.append(depth_item)
                    else:
                        self.get_logger().warn(f"深度数据格式错误: {type(depth_item)}, 长度: {len(depth_item) if hasattr(depth_item, '__len__') else 'N/A'}")
                except queue.Empty:
                    break
                except Exception as e:
                    self.get_logger().warn(f"深度数据获取异常: {str(e)}")
                    break
            
            # 获取位姿数据（最多收集5帧）
            for _ in range(100):
                try:
                    pose_item = self.pose_queue.get(timeout=0.001)
                    # 验证数据格式
                    if isinstance(pose_item, tuple) and len(pose_item) == 2:
                        pose_candidates.append(pose_item)
                    else:
                        self.get_logger().warn(f"位姿数据格式错误: {type(pose_item)}, 长度: {len(pose_item) if hasattr(pose_item, '__len__') else 'N/A'}")
                except queue.Empty:
                    break
                except Exception as e:
                    self.get_logger().warn(f"位姿数据获取异常: {str(e)}")
                    break
            
            # 检查是否有足够的数据
            if not rgb_candidates or not depth_candidates or not pose_candidates:
                # 将所有数据放回队列
                self._return_all_data(rgb_candidates, depth_candidates, pose_candidates)
                return
            
            # 2. 智能匹配算法
            best_match = None
            best_score = float('inf')
            used_indices = {'rgb': -1, 'depth': -1, 'pose': -1}
            
            for i, rgb_item in enumerate(rgb_candidates):
                try:
                    rgb_timestamp, rgb_data = rgb_item
                    if rgb_data is None:
                        continue
                        
                    for j, depth_item in enumerate(depth_candidates):
                        try:
                            depth_timestamp, depth_data = depth_item
                            if depth_data is None:
                                continue
                                
                            # 检查RGB和深度是否匹配
                            rgb_depth_diff = abs(rgb_timestamp - depth_timestamp)
                            if rgb_depth_diff > MAX_RGB_DEPTH_DIFF:
                                continue
                            
                            # 计算图像平均时间戳
                            image_timestamp = (rgb_timestamp + depth_timestamp) // 2
                            
                            # 寻找最佳位姿匹配
                            for k, pose_item in enumerate(pose_candidates):
                                try:
                                    pose_timestamp, pose_matrix = pose_item
                                    if pose_matrix is None:
                                        continue
                                        
                                    pose_diff = abs(pose_timestamp - image_timestamp)
                                    if pose_diff > MAX_POSE_DIFF:
                                        continue
                                    
                                    # 计算总体匹配分数
                                    total_score = rgb_depth_diff * 2 + pose_diff
                                    
                                    if total_score < best_score:
                                        best_score = total_score
                                        best_match = {
                                            'rgb_timestamp': rgb_timestamp,
                                            'rgb_data': rgb_data,
                                            'depth_timestamp': depth_timestamp,
                                            'depth_data': depth_data,
                                            'pose_timestamp': pose_timestamp,
                                            'pose_matrix': pose_matrix,
                                            'rgb_depth_diff': rgb_depth_diff,
                                            'pose_diff': pose_diff,
                                            'image_timestamp': image_timestamp
                                        }
                                        used_indices = {'rgb': i, 'depth': j, 'pose': k}
                                except (ValueError, TypeError) as e:
                                    self.get_logger().warn(f"位姿数据解析错误: {str(e)}")
                                    continue
                        except (ValueError, TypeError) as e:
                            self.get_logger().warn(f"深度数据解析错误: {str(e)}")
                            continue
                except (ValueError, TypeError) as e:
                    self.get_logger().warn(f"RGB数据解析错误: {str(e)}")
                    continue
            
            # 3. 处理匹配结果
            if best_match is None:
                self.get_logger().warn("未找到合适的三路数据匹配")
                self._discard_old_data_smart(rgb_candidates, depth_candidates, pose_candidates)
                return
            
            # 4. 将未使用的数据放回队列
            self._return_unused_data_optimized(rgb_candidates, depth_candidates, pose_candidates, used_indices)
            
            # 5. 处理匹配到的数据
            try:
                rgb_data = cv2.resize(best_match['rgb_data'], (640, 480))
                # depth_data = cv2.resize(best_match['depth_data'], (640, 480), interpolation=cv2.INTER_NEAREST)
                depth_data = best_match['depth_data']
                pose_matrix = best_match['pose_matrix']
                
                global first_frame_pose
                
                if first_frame_pose is None:
                    first_frame_pose = best_match['pose_matrix']
                # 6. 同步质量监控
                sync_quality = {
                    'rgb_depth_diff_ms': best_match['rgb_depth_diff'] / 1e6,
                    'pose_image_diff_ms': best_match['pose_diff'] / 1e6,
                    'total_score_ms': best_score / 1e6,
                    'rgb_timestamp': best_match['rgb_timestamp'],
                    'depth_timestamp': best_match['depth_timestamp'],
                    'pose_timestamp': best_match['pose_timestamp']
                }
                
                # 7. 性能监控
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_log_time >= 2.0:
                    fps = self.frame_count / (current_time - self.last_log_time)
                    
                    self.get_logger().info(
                        f"同步状态 - FPS: {fps:.1f}, "
                        f"RGB-Depth差异: {sync_quality['rgb_depth_diff_ms']:.1f}ms, "
                        f"位姿差异: {sync_quality['pose_image_diff_ms']:.1f}ms"
                    )
                    self.frame_count = 0
                    self.last_log_time = current_time
                
                # 8. 调用外部处理函数
                if self.data_callback:
                    self.data_callback(rgb_data, depth_data, pose_matrix, sync_quality)
                else:
                    # 兼容原有接口
                    self.process_fusion_data(rgb_data, depth_data, pose_matrix)
                    
            except Exception as e:
                self.get_logger().error(f"数据处理错误: {str(e)}")
            
        except Exception as e:
            self.get_logger().error(f"同步处理错误: {str(e)}")
            import traceback
            self.get_logger().error(f"错误堆栈: {traceback.format_exc()}")

    def _return_all_data(self, rgb_candidates, depth_candidates, pose_candidates):
        """将所有数据放回队列"""
        for item in rgb_candidates:
            try:
                if isinstance(item, tuple) and len(item) == 2:
                    self.rgb_queue.put(item, timeout=0.001)
            except (queue.Full, Exception):
                pass
        
        for item in depth_candidates:
            try:
                if isinstance(item, tuple) and len(item) == 2:
                    self.depth_queue.put(item, timeout=0.001)
            except (queue.Full, Exception):
                pass
        
        for item in pose_candidates:
            try:
                if isinstance(item, tuple) and len(item) == 2:
                    self.pose_queue.put(item, timeout=0.001)
            except (queue.Full, Exception):
                pass

    def _discard_old_data_smart(self, rgb_candidates, depth_candidates, pose_candidates):
        """智能丢弃策略：保留最新数据，丢弃过旧数据"""
        try:
            current_ns = time.time_ns()
            KEEP_THRESHOLD = 200 * 1e6  # 保留100ms内的数据
            
            # 只保留较新的RGB数据
            if rgb_candidates:
                recent_rgb = [item for item in rgb_candidates 
                            if isinstance(item, tuple) and len(item) == 2 and 
                            abs(current_ns - item[0]) < KEEP_THRESHOLD]
                if recent_rgb:
                    recent_rgb.sort(key=lambda x: x[0], reverse=True)
                    try:
                        self.rgb_queue.put(recent_rgb[0], timeout=0.001)
                    except queue.Full:
                        pass
            
            # 只保留较新的深度数据
            if depth_candidates:
                recent_depth = [item for item in depth_candidates 
                                if isinstance(item, tuple) and len(item) == 2 and 
                                abs(current_ns - item[0]) < KEEP_THRESHOLD]
                if recent_depth:
                    recent_depth.sort(key=lambda x: x[0], reverse=True)
                    try:
                        self.depth_queue.put(recent_depth[0], timeout=0.001)
                    except queue.Full:
                        pass
            
            # 保留较新的位姿数据
            if pose_candidates:
                recent_pose = [item for item in pose_candidates 
                            if isinstance(item, tuple) and len(item) == 2 and 
                            abs(current_ns - item[0]) < KEEP_THRESHOLD]
                if recent_pose:
                    recent_pose.sort(key=lambda x: x[0], reverse=True)
                    for item in recent_pose[:2]:
                        try:
                            self.pose_queue.put(item, timeout=0.001)
                        except queue.Full:
                            break
        except Exception as e:
            self.get_logger().warn(f"智能丢弃数据时出错: {str(e)}")

    def process_fusion_data(self, rgb, depth, pose):
        """
        多传感器数据融合处理[1,3](@ref)
        
        参数:
            rgb: [H, W, 3] numpy数组 (dtype=uint8)
            depth: [H, W] numpy数组 (dtype=uint16)
            pose: [4, 4] numpy数组 (dtype=float64)
        """
        # 示例：提取位姿信息
        position = pose[:3, 3]
        rotation = pose[:3, :3]
        # 此处添加实际点云生成逻辑...
        
        # 示例：打印基本信息
        # self.get_logger().info(
        #     f"RGB图像尺寸: {rgb.shape[0]}x{rgb.shape[1]} | 深度图像尺寸: {depth.shape[0]}x{depth.shape[1]}\n 融合处理: 位置=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) | "
        #     f"旋转矩阵:\n{rotation}"
        # )
        # 新增：将数据放入队列
        try:
            self.result_queue.put({
                'rgb': rgb,
                'depth': depth,
                'pose': pose
            }, timeout=0.001)
        except queue.Full:
            self.get_logger().warn("结果队列已满，丢弃数据")

    def get_synced_data(self, timeout=1.0):
        """外部获取同步数据接口"""
        return self.result_queue.get(timeout=timeout)

    def _return_unused_data_optimized(self, rgb_candidates, depth_candidates, pose_candidates, used_indices):
        """优化的数据返回策略"""
        # 返回未使用的RGB数据（只保留最新的1个）
        unused_rgb = [item for i, item in enumerate(rgb_candidates) 
                    if i != used_indices['rgb']]
        if unused_rgb:
            unused_rgb.sort(key=lambda x: x[0], reverse=True)
            try:
                self.rgb_queue.put(unused_rgb[0], timeout=0.001)
            except queue.Full:
                pass
        
        # 返回未使用的深度数据（只保留最新的1个）
        unused_depth = [item for i, item in enumerate(depth_candidates) 
                        if i != used_indices['depth']]
        if unused_depth:
            unused_depth.sort(key=lambda x: x[0], reverse=True)
            try:
                self.depth_queue.put(unused_depth[0], timeout=0.001)
            except queue.Full:
                pass
        
        # 返回未使用的位姿数据（保留最新的2个）
        unused_pose = [item for i, item in enumerate(pose_candidates) 
                    if i != used_indices['pose']]
        if unused_pose:
            unused_pose.sort(key=lambda x: x[0], reverse=True)
            for item in unused_pose[:2]:
                try:
                    self.pose_queue.put(item, timeout=0.001)
                except queue.Full:
                    break

class ROSDataset(IterableDataset):
    def __init__(self, cfg, has_depth=True):
        super(ROSDataset, self).__init__()

        self.load_arkit_depth = False
        self.start = cfg['data']['start']

        self.basedir = cfg['data']['datadir']

        # self.img_files = sorted(glob.glob(os.path.join(
        #     self.basedir, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        # self.depth_paths = sorted(
        #     glob.glob(os.path.join(
        #     self.basedir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        # self.load_poses(os.path.join(self.basedir, 'pose'))
        
        # self.img_files=self.img_files[self.start:]
        # self.depth_paths=self.depth_paths[self.start:]
        # self.poses=self.poses[self.start:]

        # self.frame_ids = range(0, len(self.img_files))
        self.num_frames = 10000000000000000 #len(self.frame_ids)
        self.cfg = cfg
        self.img_height = cfg['cam']['H']
        self.img_width = cfg['cam']['W']
        self.K = np.array([[cfg['cam']['fx'], 0.0, cfg['cam']['cx']],
                            [0.0, cfg['cam']['fy'], cfg['cam']['cy']],
                            [0.0,0.0,1.0]])
        self.fx = cfg['cam']['fx']
        self.fy = cfg['cam']['fy']
        self.cx = cfg['cam']['cx']
        self.cy = cfg['cam']['cy']
        self.depth_scale = cfg['cam']['png_depth_scale']
        self.has_depth = has_depth
        # pattern = r'scene\d{4}_\d{2}'  # \d{4}匹配4位数字，\d{2}匹配2位数字
        # matches = re.findall(pattern, cfg['data']['datadir'])
        self.video_id = 'ros' #matches


        #ROS INITIALIZATION
        rclpy.init(args=None)
    
        self.node = MultiSensorFusion()
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        
        self.first_pose = None
        self.gravity_direction = None
        self.is_initialized = False
        # 启动执行器线程
        self.spin_thread = threading.Thread(target=self.executor.spin)
        self.spin_thread.start()

    def __len__(self):
        return 100000000
    
    def initialize_gravity_reference(self, first_pose):
        """
        使用第一帧的相机位姿定义重力系
        
        Args:
            first_pose: 4x4 numpy数组，第一帧的相机位姿矩阵
        """
        self.reference_pose = first_pose.copy()
        
        # 从第一帧位姿提取重力方向
        # 假设第一帧时相机基本水平，Y轴向下为重力方向
        R_first = first_pose[:3, :3]  # 提取旋转矩阵
        
        # 相机坐标系的Y轴在世界坐标系中的方向
        camera_y_in_world = R_first @ np.array([0.0, 1.0, 0.0])
        
        # 这就是我们定义的重力方向
        self.gravity_direction = camera_y_in_world / np.linalg.norm(camera_y_in_world)
        
        # 计算基准重力对齐变换（第一帧应该是单位矩阵）
        self.gravity_transform = self.compute_gravity_transform(first_pose)
        self.is_initialized = True
        
        print(f"重力系已初始化:")
        print(f"参考位姿:\n{self.reference_pose}")
        print(f"重力方向: {self.gravity_direction}")

        
    def compute_gravity_transform(self, current_pose):
        """
        计算当前相机位姿到重力对齐的变换
        
        Args:
            current_pose: 4x4当前相机位姿矩阵
            
        Returns:
            3x3重力对齐变换矩阵
        """
        if not self.is_initialized:
            return np.eye(3)
            
        # 当前相机的旋转矩阵
        R_current = current_pose[:3, :3]
        
        # 当前相机Y轴在世界坐标系中的方向
        current_camera_y_in_world = R_current @ np.array([0.0, 1.0, 0.0])
        
        # 计算从当前相机Y轴到参考重力方向的旋转
        return self.align_vectors_to_gravity(current_camera_y_in_world)
    
    def align_vectors_to_gravity(self, current_y_direction):
        """
        计算将当前Y轴方向对齐到重力方向的变换矩阵
        """
        from scipy.spatial.transform import Rotation
        
        # 归一化当前方向
        current_y_norm = current_y_direction / np.linalg.norm(current_y_direction)
        
        # 计算旋转轴
        rotation_axis = np.cross(current_y_norm, self.gravity_direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:
            # 已经对齐或完全相反
            dot_product = np.dot(current_y_norm, self.gravity_direction)
            if dot_product > 0.999:
                return np.eye(3)  # 已对齐
            else:
                # 180度旋转，选择一个垂直轴
                if abs(current_y_norm[0]) < 0.9:
                    perp_axis = np.array([1.0, 0.0, 0.0])
                else:
                    perp_axis = np.array([0.0, 0.0, 1.0])
                rotation = Rotation.from_rotvec(np.pi * perp_axis)
                return rotation.as_matrix()
        
        # 计算旋转角度
        rotation_axis = rotation_axis / rotation_axis_norm
        cos_angle = np.dot(current_y_norm, self.gravity_direction)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        print(angle)
        
        # 使用轴角表示计算旋转矩阵
        rotation = Rotation.from_rotvec(angle * rotation_axis)
        return rotation.as_matrix()

    def __iter__(self):
        print("Waiting for frames...")
        video_id = self.video_id
        index = 0
        #start ROS Loop
        while True:
            loop_start_time = time.time()
            
            try:
                # 1. 获取同步数据
                data_start_time = time.time()
                data = self.node.get_synced_data(timeout=1.0)
                data_end_time = time.time()
                print(f"[TIMING] 获取同步数据耗时: {(data_end_time - data_start_time)*1000:.2f}ms")
                
                color_data = data['rgb']
                depth_data = data['depth']
                pose = data['pose']

                # 2. 图像预处理
                preprocess_start_time = time.time()
                color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
                # print("depth_data:",depth_data,depth_data.shape)
                depth_data = depth_data.astype(np.float32) / self.depth_scale
                
                H, W = self.img_height,self.img_width#depth_data.shape
                color_data = cv2.resize(color_data, (W, H))
                preprocess_end_time = time.time()
                print(f"[TIMING] 图像预处理耗时: {(preprocess_end_time - preprocess_start_time)*1000:.2f}ms")

                # 3. 数据结构初始化
                struct_start_time = time.time()
                result = dict(wide=dict())
                wide = PosedSensorInfo()            
                
                # OK, we have a frame. Fill on the requisite data/fields.
                image_info = ImageMeasurementInfo(
                    size=(self.img_width, self.img_height),
                    K=torch.tensor([
                        [self.fx, 0.0, self.cx],
                        [0.0, self.fy, self.cy],
                        [0.0, 0.0, 1.0]
                    ])[None])
                
                image = np.asarray(color_data).reshape((self.img_height, self.img_width, 3))
                wide.image = image_info
                struct_end_time = time.time()
                print(f"[TIMING] 数据结构初始化耗时: {(struct_end_time - struct_start_time)*1000:.2f}ms")

                # 4. 图像tensor转换
                tensor_start_time = time.time()
                result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]
                tensor_end_time = time.time()
                print(f"[TIMING] 图像tensor转换耗时: {(tensor_end_time - tensor_start_time)*1000:.2f}ms")

                if self.load_arkit_depth and not self.has_depth:
                    raise ValueError("Depth was not found, you likely can only run the RGB only model with your device")

                depth_info = None            
                if self.has_depth:
                    # 5. 深度处理
                    depth_start_time = time.time()
                    
                    # We'll eventually ensure this is 1/4.
                    depth_info = DepthMeasurementInfo(
                        size=(self.img_width, self.img_height),
                        K=torch.tensor([
                            [self.fx  , 0.0, self.cx ],
                            [0.0, self.fy , self.cy ],
                            [0.0, 0.0, 1.0]
                        ])[None])

                    depth_scale = self.depth_scale
                    wide.depth = depth_info

                    # If I understand this correctly, it looks like this might just want the lower 16 bits?
                    depth_resize_start = time.time()
                    # depth_data = cv2.resize(depth_data, (self.img_width, self.img_height),interpolation=cv2.INTER_NEAREST)
                    depth_resize_end = time.time()
                    print(f"[TIMING] 深度图像resize耗时: {(depth_resize_end - depth_resize_start)*1000:.2f}ms")

                    depth_tensor_start = time.time()
                    depth = torch.tensor(depth_data.view(dtype=np.float32).reshape((self.img_height, self.img_width)))[None].float()
                    result["wide"]["depth"] = depth
                    depth_tensor_end = time.time()
                    print(f"[TIMING] 深度tensor转换耗时: {(depth_tensor_end - depth_tensor_start)*1000:.2f}ms")
                    
                    # 6. 图像缩放处理
                    scale_start_time = time.time()
                    if max(wide.image.size) > MAX_LONG_SIDE:
                        scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                        new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                        wide.image = wide.image.resize(new_size)
                        
                        pil_start = time.time()
                        result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]
                        pil_end = time.time()
                        # print(f"[TIMING] PIL图像处理耗时: {(pil_end - pil_start)*1000:.2f}ms")
                    
                    # scale_end_time = time.time()
                    # print(f"[TIMING] 图像缩放处理总耗时: {(scale_end_time - scale_start_time)*1000:.2f}ms")
                    
                    # depth_end_time = time.time()
                    # print(f"[TIMING] 深度处理总耗时: {(depth_end_time - depth_start_time)*1000:.2f}ms")
                    
                else:
                    # 7. RGB-only处理
                    rgb_only_start = time.time()
                    scale_factor = 1
                    new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                    wide.image = wide.image.resize(new_size)
                    result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]
                    rgb_only_end = time.time()
                    # print(f"[TIMING] RGB-only处理耗时: {(rgb_only_end - rgb_only_start)*1000:.2f}ms")

                # 8. 位姿处理
                pose_start_time = time.time()
                RT = torch.from_numpy(pose.astype(np.float32).reshape((4, 4)))
                print(f"pose: {pose}")
                # RT = torch.tensor(
                # compute_VC2VW_from_RC2RW(np.asarray(pose).astype(np.float32).reshape((4, 4)).T))
                wide.RT = RT[None]

                current_orientation = wide.orientation
                target_orientation = ImageOrientation.UPRIGHT
                
                # print(current_orientation)
                # print(target_orientation)

                # if self.first_pose is None:
                #     global first_frame_pose
                #     self.first_pose = RT
                #     self.initialize_gravity_reference(RT.cpu().numpy())
                def extract_x_z_axis_rotations(rotation_matrix):

                    from scipy.spatial.transform import Rotation
                    
                    # 转换为numpy
                    if isinstance(rotation_matrix, torch.Tensor):
                        R = rotation_matrix.cpu().numpy()
                    else:
                        R = rotation_matrix
                    
                    # 通过欧拉角分解 (XYZ顺序)
                    rotation = Rotation.from_matrix(R)
                    angles = rotation.as_euler('xyz')  # [绕X轴, 绕Y轴, 绕Z轴]
                    
                    angle_x, angle_y, angle_z = angles
                    
                    filtered_rotation = Rotation.from_euler('xyz', [angle_x, 0.0, angle_z])
                    

                    return torch.tensor(filtered_rotation.as_matrix(), dtype=torch.float32)
                
                T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation)
                wide = wide.orient(current_orientation, target_orientation)
                # another way
                # T_gravity = torch.inverse(extract_x_z_axis_rotations(RT[:3, :3])) # self.compute_gravity_transform(RT.cpu().numpy())
                # wide = wide.orient(current_orientation, target_orientation)


                # T_gravity = extract_roll_pitch_only(T_gravity)
                # log
                print(f"T_gravity: {T_gravity}")


                wide.RT = torch.eye(4)[None]  
                wide.T_gravity = T_gravity[None]

                gt = PosedSensorInfo()        
                gt.RT = parse_transform_4x4_np(pose)[None]
                if depth_info is not None:
                    gt.depth = depth_info
                pose_end_time = time.time()
                print(f"[TIMING] 位姿处理耗时: {(pose_end_time - pose_start_time)*1000:.2f}ms")

                # 9. 最终结果组装
                final_start_time = time.time()
                sensor_info = SensorArrayInfo()
                sensor_info.wide = wide
                sensor_info.gt = gt

                result["meta"] = dict(video_id=video_id, timestamp=index)
                result["sensor_info"] = sensor_info
                final_end_time = time.time()
                print(f"[TIMING] 最终结果组装耗时: {(final_end_time - final_start_time)*1000:.2f}ms")

                # 10. 总体时间统计
                loop_end_time = time.time()
                total_time = (loop_end_time - loop_start_time) * 1000
                print(f"[TIMING] ========== 单帧总处理时间: {total_time:.2f}ms ==========")
                print(f"[TIMING] 处理帧率: {1000/total_time:.1f} FPS")
                print("")

                index += 1
                yield result

            except queue.Empty:
                empty_time = time.time()
                print(f"[TIMING] 队列为空，等待数据... 时间: {time.strftime('%H:%M:%S', time.localtime(empty_time))}")
                time.sleep(0.1)
            except Exception as e:
                error_time = time.time()
                print(f"[TIMING] 处理异常: {str(e)} 时间: {time.strftime('%H:%M:%S', time.localtime(error_time))}")
                import traceback
                traceback.print_exc()

if __name__ == '__main__':
    config = './config/online.yaml'
    with open(config, 'r') as  f:
        cfg = yaml.full_load(f)
    ros_dataset = ROSDataset(cfg)
    for sample in ros_dataset:
        print("sample_video_id",sample['sensor_info'].gt.RT)

