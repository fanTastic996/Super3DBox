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
import random

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

# Acts like CubifyAnythingDataset but reads from the NeRFCapture stream.
class CaptureDataset(IterableDataset):
    def __init__(self, load_arkit_depth=True):
        super(CaptureDataset, self).__init__()

        self.load_arkit_depth = load_arkit_depth
        
        self.domain = Domain(domain_id=0, config=dds_config)
        self.participant = DomainParticipant()
        self.qos = Qos(Policy.Reliability.Reliable(
            max_blocking_time=duration(seconds=1)))
        self.topic = Topic(self.participant, "Frames", CaptureFrame, qos=self.qos)
        self.reader = DataReader(self.participant, self.topic)

    def __iter__(self):
        print("Waiting for frames...")
        video_id = 0

        # Start DDS Loop
        while True:
            sample = self.reader.read_next()
            if not sample:
                print("Still waiting...")
                time.sleep(0.05)
                continue

            result = dict(wide=dict())
            wide = PosedSensorInfo()            
            
            # OK, we have a frame. Fill on the requisite data/fields.
            image_info = ImageMeasurementInfo(
                size=(sample.width, sample.height),
                K=torch.tensor([
                    [sample.fl_x, 0.0, sample.cx],
                    [0.0, sample.fl_y, sample.cy],
                    [0.0, 0.0, 1.0]
                ])[None])

            print("image_info.size",image_info.size,image_info.K)

            image = np.asarray(sample.image, dtype=np.uint8).reshape((sample.height, sample.width, 3))
            wide.image = image_info
            result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]

            if self.load_arkit_depth and not sample.has_depth:
                raise ValueError("Depth was not found, you likely can only run the RGB only model with your device")
            
            depth_info = None            
            if sample.has_depth:
                # We'll eventually ensure this is 1/4.
                rgb_depth_ratio = sample.width / sample.depth_width
                depth_info = DepthMeasurementInfo(
                    size=(sample.depth_width, sample.depth_height),
                    K=torch.tensor([
                        [sample.fl_x / rgb_depth_ratio , 0.0, sample.cx / rgb_depth_ratio],
                        [0.0, sample.fl_y / rgb_depth_ratio, sample.cy / rgb_depth_ratio],
                        [0.0, 0.0, 1.0]
                    ])[None])

                # Is this an encoding thing?
                depth_scale = sample.depth_scale
                print(depth_scale)
                wide.depth = depth_info

                # If I understand this correctly, it looks like this might just want the lower 16 bits?
                depth = torch.tensor(
                    np.asarray(sample.depth_image, dtype=np.uint8).view(dtype=np.float32).reshape((sample.depth_height, sample.depth_width)))[None].float()
                result["wide"]["depth"] = depth
                
                desired_image_size = (4 * depth_info.size[0], 4 * depth_info.size[1])
                wide.image = wide.image.resize(desired_image_size)
                result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(desired_image_size)), -1, 0))[None]
            else:
                # Even for RGB-only, only support a certain long size.
                if max(wide.image.size) > MAX_LONG_SIDE:
                    scale_factor = MAX_LONG_SIDE / max(wide.image.size)

                    new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                    wide.image = wide.image.resize(new_size)
                    result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]

            # ARKit sends W2C?
            # While we don't necessarily care about pose, we use it to derive the orientation
            # and T_gravity.
            print("transform_matrix",sample.transform_matrix)
            RT = torch.tensor(
                compute_VC2VW_from_RC2RW(np.asarray(sample.transform_matrix).astype(np.float32).reshape((4, 4)).T))
            wide.RT = RT[None]
            print("wide.orientation",wide.orientation)
            current_orientation = wide.orientation
            target_orientation = ImageOrientation.UPRIGHT

            T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation)
            wide = wide.orient(current_orientation, target_orientation)

            result["wide"]["image"] = rotate_tensor(result["wide"]["image"], current_orientation, target=target_orientation)
            if wide.has("depth"):
                result["wide"]["depth"] = rotate_tensor(result["wide"]["depth"], current_orientation, target=target_orientation)

            # No need for pose anymore.
            wide.RT = torch.eye(4)[None]
            wide.T_gravity = T_gravity[None]

            sensor_info = SensorArrayInfo()
            sensor_info.wide = wide
                            
            result["meta"] = dict(video_id=video_id, timestamp=sample.timestamp)
            result["sensor_info"] = sensor_info
            #'wide': {'image'} [1, 3, 768, 1024] [1,3,H,W] torch.uint8
            #'meta': {'video_id': 0, 'timestamp': 29238.765704083}
            #'sensor_info': wide
            print("T_gravity",wide.T_gravity)

            print("wide image",result['wide']['image'].shape)
            print("meta",result['meta'])
            print("image",wide.image.size) #image (1024, 768)
            print("image",wide.image.K) #image (1024, 768)

            yield result


class ScannetDataset(IterableDataset):
    def __init__(self, cfg, has_depth=True):
        super(ScannetDataset, self).__init__()

        self.load_arkit_depth = False
        self.start = cfg['data']['start']

        self.basedir = cfg['data']['datadir']

        self.img_files = sorted(glob.glob(os.path.join(
            self.basedir, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(
            glob.glob(os.path.join(
            self.basedir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(os.path.join(self.basedir, 'pose'))
        
        self.img_files=self.img_files[self.start:]
        self.depth_paths=self.depth_paths[self.start:]
        self.poses=self.poses[self.start:]

        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)
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
        pattern = r'scene\d{4}_\d{2}'  # \d{4}匹配4位数字，\d{2}匹配2位数字
        matches = re.findall(pattern, cfg['data']['datadir'])
        self.video_id = matches

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        self.last_valid_pose = None
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)

            if not np.isinf(c2w).any():
                self.last_valid_pose = c2w
            else:
                c2w = self.last_valid_pose 
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            # c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

    def __len__(self):
        return self.num_frames



    def __iter__(self):
        print("Waiting for frames...")
        video_id = self.video_id
        index = 0
        while True:

            #Step1: load data
            color_path = self.img_files[index]
            depth_path = self.depth_paths[index]
            color_data = cv2.imread(color_path)

            if '.png' in depth_path:
                depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            elif '.exr' in depth_path:
                raise NotImplementedError()

            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            color_data = color_data 
            depth_data = depth_data.astype(np.float32) / self.depth_scale #* self.sc_factor

            H, W = depth_data.shape
            color_data = cv2.resize(color_data, (W, H))
            pose = self.poses[index]

            #Step2:try to warp the data like the original dataset    
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

            # print(image_info.size)

            image = np.asarray(color_data).reshape((self.img_height, self.img_width, 3))

            wide.image = image_info
            result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]

            if self.load_arkit_depth and not self.has_depth:
                raise ValueError("Depth was not found, you likely can only run the RGB only model with your device")

            depth_info = None            
            if self.has_depth:
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
                depth_data = cv2.resize(depth_data, (self.img_width, self.img_height))

                depth = torch.tensor(depth_data.view(dtype=np.float32).reshape((self.img_height, self.img_width)))[None].float()
                result["wide"]["depth"] = depth
                
                # desired_image_size = (4 * depth_info.size[0], 4 * depth_info.size[1])
                # wide.image = wide.image.resize(desired_image_size)

                if max(wide.image.size) > MAX_LONG_SIDE:
                    scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                    # scale_factor = 1
                    new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                    wide.image = wide.image.resize(new_size)
                    result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]
                
            else:
                # Even for RGB-only, only support a certain long size.
                # if max(wide.image.size) > MAX_LONG_SIDE:
                # scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                scale_factor = 1

                new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                wide.image = wide.image.resize(new_size)
                result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]


            # ARKit sends W2C?
            # While we don't necessarily care about pose, we use it to derive the orientation
            # and T_gravity.
            RT = torch.from_numpy(pose.astype(np.float32).reshape((4, 4)))
            wide.RT = RT[None]

            current_orientation = wide.orientation
            target_orientation = ImageOrientation.UPRIGHT

            T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation)
            wide = wide.orient(current_orientation, target_orientation)

            '''
            Rotate IMG and Depth
            '''
            result["wide"]["image"] = rotate_tensor(result["wide"]["image"], current_orientation, target=target_orientation)
            if wide.has("depth"):
                result["wide"]["depth"] = rotate_tensor(result["wide"]["depth"], current_orientation, target=target_orientation)

            # No need for pose anymore.
            wide.RT = torch.eye(4)[None]
            wide.T_gravity = T_gravity[None]
            print(f"T_gravity: {T_gravity}")

            gt = PosedSensorInfo()        
            gt.RT = parse_transform_4x4_np(pose)[None]
            if depth_info is not None:
                gt.depth = depth_info

            sensor_info = SensorArrayInfo()
            sensor_info.wide = wide
            sensor_info.gt = gt

            result["meta"] = dict(video_id=video_id, timestamp=index)
            result["sensor_info"] = sensor_info

            # print("wide image",result['wide']['image'].shape)
            # print("meta",result['meta'])
            # print("image",wide.image.size) #image (1024, 768)

            index+=1
            yield result

class LargeDataset(IterableDataset):
    def __init__(self, cfg, has_depth=True):
        super(LargeDataset, self).__init__()

        self.load_arkit_depth = False
        self.start = cfg['data']['start']

        self.basedir = cfg['data']['datadir']

        self.img_files = sorted(glob.glob(os.path.join(
            self.basedir, 'color', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(
            glob.glob(os.path.join(
            self.basedir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        # self.load_poses(os.path.join(self.basedir, 'pose'))
        self.poses = np.load(os.path.join(self.basedir,'all_poses_RO.npy')).reshape(-1,4,4)
        
        self.img_files=self.img_files[self.start:]
        self.depth_paths=self.depth_paths[self.start:]
        self.poses=self.poses[self.start:]

        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)
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

        self.video_id = 'building2'

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        self.last_valid_pose = None
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)

            if not np.isinf(c2w).any():
                self.last_valid_pose = c2w
            else:
                c2w = self.last_valid_pose 
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            # c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

    def __len__(self):
        return self.num_frames



    def __iter__(self):
        print("Waiting for frames...")
        video_id = self.video_id
        index = 0
        while True:

            #Step1: load data
            color_path = self.img_files[index]
            depth_path = self.depth_paths[index]
            color_data = cv2.imread(color_path)

            if '.png' in depth_path:
                depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            elif '.exr' in depth_path:
                raise NotImplementedError()

            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            color_data = color_data 
            depth_data = depth_data.astype(np.float32) / self.depth_scale #* self.sc_factor

            H, W = depth_data.shape
            color_data = cv2.resize(color_data, (W, H))
            pose = self.poses[index]

            #Step2:try to warp the data like the original dataset    
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

            # print(image_info.size)

            image = np.asarray(color_data).reshape((self.img_height, self.img_width, 3))

            wide.image = image_info
            result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]

            if self.load_arkit_depth and not self.has_depth:
                raise ValueError("Depth was not found, you likely can only run the RGB only model with your device")

            depth_info = None            
            if self.has_depth:
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
                depth_data = cv2.resize(depth_data, (self.img_width, self.img_height))

                depth = torch.tensor(depth_data.view(dtype=np.float32).reshape((self.img_height, self.img_width)))[None].float()
                result["wide"]["depth"] = depth
                
                # desired_image_size = (4 * depth_info.size[0], 4 * depth_info.size[1])
                # wide.image = wide.image.resize(desired_image_size)

                if max(wide.image.size) > MAX_LONG_SIDE:
                    scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                    # scale_factor = 1
                    new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                    wide.image = wide.image.resize(new_size)
                    result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]
                
            else:
                # Even for RGB-only, only support a certain long size.
                # if max(wide.image.size) > MAX_LONG_SIDE:
                # scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                scale_factor = 1

                new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                wide.image = wide.image.resize(new_size)
                result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]


            # ARKit sends W2C?
            # While we don't necessarily care about pose, we use it to derive the orientation
            # and T_gravity.
            RT = torch.from_numpy(pose.astype(np.float32).reshape((4, 4)))
            wide.RT = RT[None]

            current_orientation = wide.orientation
            target_orientation = ImageOrientation.UPRIGHT


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
                
                # filtered_rotation = Rotation.from_euler('xyz', [angle_x, 0.0, angle_z])
                filtered_rotation = Rotation.from_euler('xyz', [0.0, angle_y, angle_z]) 
                

                return torch.tensor(filtered_rotation.as_matrix(), dtype=torch.float32)

            T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation)
            # T_gravity = torch.inverse(extract_x_z_axis_rotations(RT[:3, :3]))
            wide = wide.orient(current_orientation, target_orientation)

            '''
            Rotate IMG and Depth
            '''
            result["wide"]["image"] = rotate_tensor(result["wide"]["image"], current_orientation, target=target_orientation)
            if wide.has("depth"):
                result["wide"]["depth"] = rotate_tensor(result["wide"]["depth"], current_orientation, target=target_orientation)

            # No need for pose anymore.
            wide.RT = torch.eye(4)[None]
            wide.T_gravity = T_gravity[None]


            gt = PosedSensorInfo()        
            gt.RT = parse_transform_4x4_np(pose)[None]
            if depth_info is not None:
                gt.depth = depth_info

            sensor_info = SensorArrayInfo()
            sensor_info.wide = wide
            sensor_info.gt = gt

            result["meta"] = dict(video_id=video_id, timestamp=index)
            result["sensor_info"] = sensor_info

            # print("wide image",result['wide']['image'].shape)
            # print("meta",result['meta'])
            # print("image",wide.image.size) #image (1024, 768)

            index+=1
            yield result


# class CA1MDataset(IterableDataset):
#     def __init__(self, cfg, has_depth=True):
#         super(CA1MDataset, self).__init__()

#         self.load_arkit_depth = False
#         self.start = cfg['data']['start']

#         self.basedir = cfg['data']['datadir']
#         self.img_files = sorted(glob.glob(os.path.join(
#             self.basedir, 'rgb', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
#         self.depth_paths = sorted(
#             glob.glob(os.path.join(
#             self.basedir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
#         self.load_poses(os.path.join(self.basedir, 'all_poses.npy'))
        
#         self.img_files=self.img_files[self.start:]
#         self.depth_paths=self.depth_paths[self.start:]
#         self.poses=self.poses[self.start:]

#         self.frame_ids = range(0, len(self.img_files))
#         self.num_frames = len(self.frame_ids)
#         self.cfg = cfg




#         depth_intric = np.loadtxt(os.path.join(self.basedir, 'K_depth.txt')).reshape(3,3)
#         self.d_K = np.array([[depth_intric[0,0], 0.0, depth_intric[0,2]],
#                             [0.0, depth_intric[1,1], depth_intric[1,2]],
#                             [0.0,0.0,1.0]])
#         self.dfx = self.d_K[0,0]
#         self.dfy = self.d_K[1,1]
#         self.dcx = self.d_K[0,2]
#         self.dcy = self.d_K[1,2]

#         rgb_intric = np.loadtxt(os.path.join(self.basedir, 'K_rgb.txt')).reshape(3,3)
#         self.r_K = np.array([[rgb_intric[0,0], 0.0, rgb_intric[0,2]],
#                             [0.0, rgb_intric[1,1], rgb_intric[1,2]],
#                             [0.0,0.0,1.0]])
#         self.rfx = self.r_K[0,0]
#         self.rfy = self.r_K[1,1]
#         self.rcx = self.r_K[0,2]
#         self.rcy = self.r_K[1,2]


#         if self.r_K[0,2]< self.r_K[1,2]:
#             self.img_height=cfg["cam"]["W"] #l
#             self.img_width=cfg["cam"]["H"] #s
#         else:
#             self.img_height=cfg["cam"]["H"]
#             self.img_width=cfg["cam"]["W"]


#         self.depth_scale = cfg['cam']['png_depth_scale']
#         self.has_depth = has_depth
#         pattern = r'\b4\d{7}\b'  
#         matches = re.findall(pattern, cfg['data']['datadir'])
#         self.video_id = matches


#     def load_poses(self, path):
#         self.poses = np.load(path).reshape(-1,4,4)

#     def __len__(self):
#         return self.num_frames



#     def __iter__(self):
#         print("Waiting for frames...")
#         video_id = self.video_id
#         index = 0
#         while True:

#             #Step1: load data
#             color_path = self.img_files[index]
#             depth_path = self.depth_paths[index]
#             color_data = cv2.imread(color_path)

#             if '.png' in depth_path:
#                 depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
#             elif '.exr' in depth_path:
#                 raise NotImplementedError()

#             color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
#             color_data = color_data 
#             depth_data = depth_data.astype(np.float32) / self.depth_scale #* self.sc_factor

#             # H, W = depth_data.shape
#             # color_data = cv2.resize(color_data, (W, H))
#             pose = self.poses[index]

#             #Step2:try to warp the data like the original dataset    
#             result = dict(wide=dict())
#             wide = PosedSensorInfo()            
            
#             # OK, we have a frame. Fill on the requisite data/fields.
#             image_info = ImageMeasurementInfo(
#                 size=(color_data.shape[1], color_data.shape[0]),
#                 K=torch.tensor([
#                     [self.rfx, 0.0, self.rcx],
#                     [0.0, self.rfy, self.rcy],
#                     [0.0, 0.0, 1.0]
#                 ])[None])

#             image = np.asarray(color_data)#.reshape((self.img_height, self.img_width, 3))

#             wide.image = image_info
#             result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]

#             if self.load_arkit_depth and not self.has_depth:
#                 raise ValueError("Depth was not found, you likely can only run the RGB only model with your device")

#             depth_info = None            
#             if self.has_depth:
#                 # We'll eventually ensure this is 1/4.
#                 depth_info = DepthMeasurementInfo(
#                     size=(depth_data.shape[1], depth_data.shape[0]),
#                     K=torch.tensor([
#                         [self.dfx, 0.0, self.dcx ],
#                         [0.0, self.dfy, self.dcy ],
#                         [0.0, 0.0, 1.0]
#                     ])[None])

#                 depth_scale = self.depth_scale
#                 wide.depth = depth_info

#                 # If I understand this correctly, it looks like this might just want the lower 16 bits?
#                 # depth_data = cv2.resize(depth_data, (self.img_width, self.img_height))

#                 depth = torch.tensor(depth_data.view(dtype=np.float32))[None].float()
#                 result["wide"]["depth"] = depth
                
#                 # desired_image_size = (4 * depth_info.size[0], 4 * depth_info.size[1])
#                 # wide.image = wide.image.resize(desired_image_size)

#                 if max(wide.image.size) > MAX_LONG_SIDE:
#                     scale_factor = MAX_LONG_SIDE / max(wide.image.size)
#                     # scale_factor = 1
#                     new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
#                     wide.image = wide.image.resize(new_size)
#                     result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]
                
#             else:
#                 # Even for RGB-only, only support a certain long size.
#                 # if max(wide.image.size) > MAX_LONG_SIDE:
#                 # scale_factor = MAX_LONG_SIDE / max(wide.image.size)
#                 scale_factor = 1

#                 new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
#                 wide.image = wide.image.resize(new_size)
#                 result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]


#             # ARKit sends W2C?
#             # While we don't necessarily care about pose, we use it to derive the orientation
#             # and T_gravity.
#             RT = torch.from_numpy(pose.astype(np.float32).reshape((4, 4)))
#             wide.RT = RT[None]

#             current_orientation = wide.orientation
#             target_orientation = ImageOrientation.UPRIGHT

#             T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation)
#             wide = wide.orient(current_orientation, target_orientation)

#             '''
#             Rotate IMG and Depth
#             '''
#             result["wide"]["image"] = rotate_tensor(result["wide"]["image"], current_orientation, target=target_orientation)
#             if wide.has("depth"):
#                 result["wide"]["depth"] = rotate_tensor(result["wide"]["depth"], current_orientation, target=target_orientation)

#             # No need for pose anymore.
#             wide.RT = torch.eye(4)[None]
#             wide.T_gravity = T_gravity[None]


#             gt = PosedSensorInfo()        
#             gt.RT = parse_transform_4x4_np(pose)[None]
#             if depth_info is not None:
#                 gt.depth = depth_info

#             sensor_info = SensorArrayInfo()
#             sensor_info.wide = wide
#             sensor_info.gt = gt

#             result["meta"] = dict(video_id=video_id, timestamp=index)
#             result["sensor_info"] = sensor_info


#             index+=1
#             yield result

class CA1MDataset(IterableDataset):
    def __init__(self, cfg, has_depth=True):
        super(CA1MDataset, self).__init__()

        self.load_arkit_depth = False
        self.start = cfg['data']['start']

        self.basedir = cfg['data']['datadir']
        self.img_files = sorted(glob.glob(os.path.join(
            self.basedir, 'rgb', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(
            glob.glob(os.path.join(
            self.basedir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(os.path.join(self.basedir, 'all_poses.npy'))
        
        self.img_files=self.img_files[self.start:]
        self.depth_paths=self.depth_paths[self.start:]
        self.poses=self.poses[self.start:]

        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)
        self.cfg = cfg




        depth_intric = np.loadtxt(os.path.join(self.basedir, 'K_depth.txt')).reshape(3,3)
        self.K = np.array([[depth_intric[0,0], 0.0, depth_intric[0,2]],
                            [0.0, depth_intric[1,1], depth_intric[1,2]],
                            [0.0,0.0,1.0]])
        self.fx = self.K[0,0]
        self.fy = self.K[1,1]
        self.cx = self.K[0,2]
        self.cy = self.K[1,2]

        if self.K[0,2]< self.K[1,2]:
            self.img_height=cfg["cam"]["W"] #l
            self.img_width=cfg["cam"]["H"] #s
        else:
            self.img_height=cfg["cam"]["H"]
            self.img_width=cfg["cam"]["W"]


        self.depth_scale = cfg['cam']['png_depth_scale']
        self.has_depth = has_depth
        pattern = r'\b4\d{7}\b'  
        matches = re.findall(pattern, cfg['data']['datadir'])
        self.video_id = matches


    def load_poses(self, path):
        self.poses = np.load(path).reshape(-1,4,4)

    def __len__(self):
        return self.num_frames

    # #single frame iterator
    # def __iter__(self):
    #     print("Waiting for frames...")
    #     video_id = self.video_id
    #     index = 0
    #     while True:

    #         #Step1: load data
    #         color_path = self.img_files[index]
    #         depth_path = self.depth_paths[index]
    #         # print("color_path",color_path)
    #         color_data = cv2.imread(color_path)

    #         if '.png' in depth_path:
    #             depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    #         elif '.exr' in depth_path:
    #             raise NotImplementedError()
            
    #         color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
    #         color_data = color_data 
    #         depth_data = depth_data.astype(np.float32) / self.depth_scale #* self.sc_factor

    #         H, W = depth_data.shape
    #         color_data = cv2.resize(color_data, (W, H))
    #         pose = self.poses[index]

    #         #Step2:try to warp the data like the original dataset    
    #         result = dict(wide=dict())
    #         wide = PosedSensorInfo()            
            
    #         # OK, we have a frame. Fill on the requisite data/fields.
    #         image_info = ImageMeasurementInfo(
    #             size=(self.img_width, self.img_height),
    #             K=torch.tensor([
    #                 [self.fx, 0.0, self.cx],
    #                 [0.0, self.fy, self.cy],
    #                 [0.0, 0.0, 1.0]
    #             ])[None])

    #         image = np.asarray(color_data).reshape((self.img_height, self.img_width, 3))

    #         wide.image = image_info
    #         result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]

    #         if self.load_arkit_depth and not self.has_depth:
    #             raise ValueError("Depth was not found, you likely can only run the RGB only model with your device")

    #         depth_info = None            
    #         if self.has_depth:
    #             # We'll eventually ensure this is 1/4.
    #             depth_info = DepthMeasurementInfo(
    #                 size=(self.img_width, self.img_height),
    #                 K=torch.tensor([
    #                     [self.fx, 0.0, self.cx ],
    #                     [0.0, self.fy, self.cy ],
    #                     [0.0, 0.0, 1.0]
    #                 ])[None])

    #             depth_scale = self.depth_scale
    #             wide.depth = depth_info

    #             # If I understand this correctly, it looks like this might just want the lower 16 bits?
    #             depth_data = cv2.resize(depth_data, (self.img_width, self.img_height))

    #             depth = torch.tensor(depth_data.view(dtype=np.float32).reshape((self.img_height, self.img_width)))[None].float()
    #             result["wide"]["depth"] = depth
                
    #             # desired_image_size = (4 * depth_info.size[0], 4 * depth_info.size[1])
    #             # wide.image = wide.image.resize(desired_image_size)

    #             if max(wide.image.size) > MAX_LONG_SIDE:
    #                 scale_factor = MAX_LONG_SIDE / max(wide.image.size)
    #                 # scale_factor = 1
    #                 new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
    #                 wide.image = wide.image.resize(new_size)
    #                 result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]
                
    #         else:
    #             # Even for RGB-only, only support a certain long size.
    #             # if max(wide.image.size) > MAX_LONG_SIDE:
    #             # scale_factor = MAX_LONG_SIDE / max(wide.image.size)
    #             scale_factor = 1

    #             new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
    #             wide.image = wide.image.resize(new_size)
    #             result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]


    #         # ARKit sends W2C?
    #         # While we don't necessarily care about pose, we use it to derive the orientation
    #         # and T_gravity.
    #         RT = torch.from_numpy(pose.astype(np.float32).reshape((4, 4)))
    #         wide.RT = RT[None]

    #         current_orientation = wide.orientation
    #         target_orientation = ImageOrientation.UPRIGHT

    #         T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation)
    #         wide = wide.orient(current_orientation, target_orientation)

    #         '''
    #         Rotate IMG and Depth
    #         '''
    #         result["wide"]["image"] = rotate_tensor(result["wide"]["image"], current_orientation, target=target_orientation)
    #         if wide.has("depth"):
    #             result["wide"]["depth"] = rotate_tensor(result["wide"]["depth"], current_orientation, target=target_orientation)

    #         # No need for pose anymore.
    #         wide.RT = torch.eye(4)[None]
    #         wide.T_gravity = T_gravity[None]


    #         gt = PosedSensorInfo()        
    #         gt.RT = parse_transform_4x4_np(pose)[None]
    #         if depth_info is not None:
    #             gt.depth = depth_info

    #         sensor_info = SensorArrayInfo()
    #         sensor_info.wide = wide
    #         sensor_info.gt = gt

    #         result["meta"] = dict(video_id=video_id, timestamp=index)
    #         result["meta"]['img_path'] = color_path
    #         result["sensor_info"] = sensor_info
    #         index+=1
    #         yield result
    
    def generate_unique_randoms(self, n, start, end):
        unique_nums = set()  # 用集合存储不重复数字
        while len(unique_nums) < n:  # 循环直到集齐 n 个数
            num = random.randint(start, end)  # 生成随机数
            unique_nums.add(num)
        return list(unique_nums)  # 转为列表返
    
    def __iter__(self):
        print("Waiting for frames...")
        video_id = self.video_id
        
        while True:

            all_image = []
            all_depth = []
            all_timestamp = []
            all_img_path = []
            # 生成 6 个 [0, 1024] 的不重复随机数
            # random_indexs = self.generate_unique_randoms(3, 0, self.num_frames)            
            random_indexs = [0, 30, 60]
            for index in random_indexs:
                #Step1: load data
                color_path = self.img_files[index]
                depth_path = self.depth_paths[index]
                # print("color_path",color_path)
                color_data = cv2.imread(color_path)

                if '.png' in depth_path:
                    depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                elif '.exr' in depth_path:
                    raise NotImplementedError()
                
                color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
                color_data = color_data 
                depth_data = depth_data.astype(np.float32) / self.depth_scale #* self.sc_factor

                H, W = depth_data.shape
                color_data = cv2.resize(color_data, (W, H))
                pose = self.poses[index]

                #Step2:try to warp the data like the original dataset    
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
                result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]

                if self.load_arkit_depth and not self.has_depth:
                    raise ValueError("Depth was not found, you likely can only run the RGB only model with your device")

                depth_info = None            
                if self.has_depth:
                    # We'll eventually ensure this is 1/4.
                    depth_info = DepthMeasurementInfo(
                        size=(self.img_width, self.img_height),
                        K=torch.tensor([
                            [self.fx, 0.0, self.cx ],
                            [0.0, self.fy, self.cy ],
                            [0.0, 0.0, 1.0]
                        ])[None])

                    depth_scale = self.depth_scale
                    wide.depth = depth_info

                    # If I understand this correctly, it looks like this might just want the lower 16 bits?
                    depth_data = cv2.resize(depth_data, (self.img_width, self.img_height))

                    depth = torch.tensor(depth_data.view(dtype=np.float32).reshape((self.img_height, self.img_width)))[None].float()
                    result["wide"]["depth"] = depth
                    
              
                    if max(wide.image.size) > MAX_LONG_SIDE:
                        scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                        # scale_factor = 1
                        new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                        wide.image = wide.image.resize(new_size)
                        result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]
                    
                else:
                    # Even for RGB-only, only support a certain long size.
                    # if max(wide.image.size) > MAX_LONG_SIDE:
                    # scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                    scale_factor = 1

                    new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                    wide.image = wide.image.resize(new_size)
                    result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]


                # ARKit sends W2C?
                # While we don't necessarily care about pose, we use it to derive the orientation
                # and T_gravity.
                RT = torch.from_numpy(pose.astype(np.float32).reshape((4, 4)))
                wide.RT = RT[None]

                current_orientation = wide.orientation
                target_orientation = ImageOrientation.UPRIGHT

                T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation)
                wide = wide.orient(current_orientation, target_orientation)

                '''
                Rotate IMG and Depth
                '''
                result["wide"]["image"] = rotate_tensor(result["wide"]["image"], current_orientation, target=target_orientation)
                if wide.has("depth"):
                    result["wide"]["depth"] = rotate_tensor(result["wide"]["depth"], current_orientation, target=target_orientation)

                # No need for pose anymore.
                wide.RT = torch.eye(4)[None]
                wide.T_gravity = T_gravity[None]


                gt = PosedSensorInfo()        
                gt.RT = parse_transform_4x4_np(pose)[None]
                if depth_info is not None:
                    gt.depth = depth_info

                sensor_info = SensorArrayInfo()
                sensor_info.wide = wide
                sensor_info.gt = gt

                result["meta"] = dict(video_id=video_id, timestamp=index)
                result["meta"]['img_path'] = color_path
                result["sensor_info"] = sensor_info
                #dict_keys(['wide', 'meta', 'sensor_info'])
                all_image.append(result["wide"]["image"])
                all_depth.append(result["wide"]["depth"])
                all_timestamp.append(index)
                all_img_path.append(color_path)
            
            all_image_tensor = torch.cat(all_image, dim=0)
            all_depth_tensor = torch.cat(all_depth, dim=0)
            
            batched_result = {
                "wide": {
                    "image": all_image_tensor,
                    "depth": all_depth_tensor 
                },
                "meta": {
                    "video_id": video_id,
                    "timestamp": all_timestamp,
                    "img_path": all_img_path
                },
                "sensor_info": sensor_info
            }
            
            yield batched_result        

    
            # 返回数据示例
            # {
            #     "wide": {
            #         "image": tensor([N, 3, H, W]),  # 图像张量
            #         "depth": tensor([N, H, W])   # 深度图（可选）
            #     },
            #     "meta": {
            #         "video_id": ['42446540'],
            #         "timestamp": [idx1, idx2, ...],  # 6个时间戳
            #         "img_path": [  # 新增的路径列表
            #             "/path/to/frame001.png",
            #             "/path/to/frame042.png",
            #             ... 
            #         ]
            #     },
            #     "sensor_info": [...]  # 传感器信息列表
            # }



class BS3DDataset(IterableDataset):
    def __init__(self, cfg, has_depth=True):
        super(BS3DDataset, self).__init__()

        self.load_arkit_depth = False

        self.basedir = cfg['data']['datadir']
        self.img_files = sorted(glob.glob(os.path.join(
                self.basedir, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(
                glob.glob(os.path.join(
                self.basedir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)
        self.load_poses(os.path.join(self.basedir, 'poses.txt'))
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

    def load_poses(self, path):
        if self.num_frames==1:

            self.poses = []
            poses = np.loadtxt(path, dtype=np.float32)
            # poses = poses[:,1:]
            poses = poses[1:]
            poses = poses[None]
            # for i in range(poses.shape[0]):
            # c2w = self.pose_matrix_from_quaternion(poses[i])
            c2w = self.pose_matrix_from_quaternion(poses[0])
                # c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)
        else:
            self.poses = []
            poses = np.loadtxt(path, dtype=np.float32)
            poses = poses[:,1:]
            for i in range(poses.shape[0]):
                c2w = self.pose_matrix_from_quaternion(poses[i])
                # c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def __len__(self):
        return self.num_frames

    def __iter__(self):
        print("Waiting for frames...")
        video_id = 0
        index = 0
        while True:
            if index>=self.num_frames:
                break
            #Step1: load data
            # print("img_files",self.img_files,index)

            color_path = self.img_files[index]
            depth_path = self.depth_paths[index]
            color_data = cv2.imread(color_path)

            if '.png' in depth_path:
                depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            elif '.exr' in depth_path:
                raise NotImplementedError()

            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            color_data = color_data 
            depth_data = depth_data.astype(np.float32) / self.depth_scale #* self.sc_factor

            H, W = depth_data.shape
            color_data = cv2.resize(color_data, (W, H))
            pose = self.poses[index]

            #Step2:try to warp the data like the original dataset    
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

            # print("image_info.size",image_info.size,image_info.K)

            image = np.asarray(color_data)#.reshape((self.img_height, self.img_width, 3))

            wide.image = image_info
            result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]

            if self.load_arkit_depth and not self.has_depth:
                raise ValueError("Depth was not found, you likely can only run the RGB only model with your device")

            depth_info = None            
            if self.has_depth:
                # We'll eventually ensure this is 1/4.
                depth_info = DepthMeasurementInfo(
                    size=(self.img_width, self.img_height),
                    K=torch.tensor([
                        [self.fx , 0.0, self.cx],
                        [0.0, self.fy , self.cy],
                        [0.0, 0.0, 1.0]
                    ])[None])

                wide.depth = depth_info

                # If I understand this correctly, it looks like this might just want the lower 16 bits?
                depth_data = cv2.resize(depth_data, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
                result["wide"]["depth"] = torch.tensor(np.array(Image.fromarray(depth_data)))[None]
                #RGB
                new_size = (int(self.img_width), int(self.img_height))
                wide.image = wide.image.resize(new_size)
                result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]


                if max(wide.image.size) > MAX_LONG_SIDE:
                    scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                    new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                    #Depth
                    wide.depth = wide.depth.resize(new_size)
                    # result["wide"]["depth"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(depth_data).resize(new_size)), -1, 0))[None]
                    depth_data = cv2.resize(depth_data, (new_size[0], new_size[1]), interpolation=cv2.INTER_NEAREST)
                    result["wide"]["depth"] = torch.tensor(np.array(Image.fromarray(depth_data)))[None]
                    # result["wide"]["depth"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(depth_data)), -1, 0))[None]

                    # print("wide depth",result['wide']['depth'])
                    # print("depth",wide.depth.size) #image (1024, 768)
                    # print("depth",wide.depth.K) #image (1024, 768)

                    
                    #RGB
                    wide.image = wide.image.resize(new_size)
                    result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]
            else:
                # Even for RGB-only, only support a certain long size.
                if max(wide.image.size) > MAX_LONG_SIDE:
                    scale_factor = MAX_LONG_SIDE / max(wide.image.size)

                    new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                    wide.image = wide.image.resize(new_size)
                    result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]


            # ARKit sends W2C?
            # While we don't necessarily care about pose, we use it to derive the orientation
            # and T_gravity.
            RT = torch.from_numpy(pose.astype(np.float32).reshape((4, 4)))
            wide.RT = RT[None]

            current_orientation = wide.orientation
            target_orientation = ImageOrientation.UPRIGHT

            T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation)
            wide = wide.orient(current_orientation, target_orientation)

            '''
            Rotate IMG and Depth
            '''
            result["wide"]["image"] = rotate_tensor(result["wide"]["image"], current_orientation, target=target_orientation)
            # if wide.has("depth"):
                # result["wide"]["depth"] = rotate_tensor(result["wide"]["depth"], current_orientation, target=target_orientation)

                # import matplotlib.pyplot as plt
                # plt.imshow(result["wide"]["depth"].squeeze().cpu().numpy(), cmap='jet')
                # plt.colorbar(label='Normalized Depth')
                # plt.title('Contrast Enhanced Depth Map')
                # plt.show()


            # No need for pose anymore.
            wide.RT = torch.eye(4)[None]
            wide.T_gravity = T_gravity[None]


            gt = PosedSensorInfo()        
            gt.RT = parse_transform_4x4_np(pose)[None]
            if depth_info is not None:
                gt.depth = wide.depth

            sensor_info = SensorArrayInfo()
            sensor_info.wide = wide
            sensor_info.gt = gt

            result["meta"] = dict(video_id=video_id, timestamp=index)
            result["sensor_info"] = sensor_info

            # print("wide image",result['wide']['image'].shape)
            # print("meta",result['meta'])
            # print("image",wide.image.size) #image (1024, 768)
            # print("image",wide.image.K) #image (1024, 768)

            index+=1
            yield result



class OwnDataset(IterableDataset):
    def __init__(self, cfg, has_depth=False):
        super(OwnDataset, self).__init__()

        self.load_arkit_depth = False

        self.basedir = cfg['data']['datadir']
        self.img_files = sorted(glob.glob(os.path.join(
            self.basedir, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(
            glob.glob(os.path.join(
            self.basedir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)
        # self.load_poses(os.path.join(self.basedir, 'all_poses.npy'))
        self.load_poses(os.path.join(self.basedir, 'traj_full.txt'))
        
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


    # def load_poses(self, path):
    #     self.poses  = np.load(path).reshape(-1,4,4)

    def load_poses(self, path):
        if self.num_frames==1:

            self.poses = []
            poses = np.loadtxt(path, dtype=np.float32)
            # poses = poses[:,1:]
            poses = poses[1:]
            poses = poses[None]
            # for i in range(poses.shape[0]):
            # c2w = self.pose_matrix_from_quaternion(poses[i])
            c2w = self.pose_matrix_from_quaternion(poses[0])
                # c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)
        else:
            self.poses = []
            poses = np.loadtxt(path, dtype=np.float32)
            poses = poses[:,1:]
            for i in range(poses.shape[0]):
                c2w = self.pose_matrix_from_quaternion(poses[i])
                # c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)
    
    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def __len__(self):
        return self.num_frames


    def __iter__(self):
        print("Waiting for frames...")
        video_id = 0
        index = 0
        while True:

            #Step1: load data
            color_path = self.img_files[index]
            # depth_path = self.depth_paths[index]
            color_data = cv2.imread(color_path)
            # print("color data",color_data.shape)
            # if '.png' in depth_path:
            #     depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            # elif '.exr' in depth_path:
            #     raise NotImplementedError()

            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            color_data = color_data 
            # depth_data = depth_data.astype(np.float32) / self.depth_scale #* self.sc_factor

            H, W = self.img_height,self.img_width
            color_data = cv2.resize(color_data, (W, H))
            pose = self.poses[index]

            #Step2:try to warp the data like the original dataset    
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

            # print(image_info.size)

            image = np.asarray(color_data).reshape((self.img_height, self.img_width, 3))

            wide.image = image_info
            result["wide"]["image"] = torch.tensor(np.moveaxis(image, -1, 0))[None]

            if self.load_arkit_depth and not self.has_depth:
                raise ValueError("Depth was not found, you likely can only run the RGB only model with your device")

            depth_info = None            
            if self.has_depth:
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
                depth_data = cv2.resize(depth_data, (self.img_width, self.img_height))

                depth = torch.tensor(depth_data.view(dtype=np.float32).reshape((self.img_height, self.img_width)))[None].float()
                result["wide"]["depth"] = depth
                
                # desired_image_size = (4 * depth_info.size[0], 4 * depth_info.size[1])
                # wide.image = wide.image.resize(desired_image_size)

                if max(wide.image.size) > MAX_LONG_SIDE:
                    scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                    # scale_factor = 1
                    new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                    wide.image = wide.image.resize(new_size)
                    result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]
                
            else:
                # Even for RGB-only, only support a certain long size.
                # if max(wide.image.size) > MAX_LONG_SIDE:
                # scale_factor = MAX_LONG_SIDE / max(wide.image.size)
                scale_factor = 1

                new_size = (int(wide.image.size[0] * scale_factor), int(wide.image.size[1] * scale_factor))
                wide.image = wide.image.resize(new_size)
                result["wide"]["image"] = torch.tensor(np.moveaxis(np.array(Image.fromarray(image).resize(new_size)), -1, 0))[None]


            # ARKit sends W2C?
            # While we don't necessarily care about pose, we use it to derive the orientation
            # and T_gravity.
            RT = torch.from_numpy(pose.astype(np.float32).reshape((4, 4)))
            wide.RT = RT[None]

    # UPRIGHT = 0
    # LEFT = 1
    # UPSIDE_DOWN = 2
    # RIGHT = 3
    # ORIGINAL = 4

            current_orientation = wide.orientation
            print("current_orientation",current_orientation)
            target_orientation = ImageOrientation.UPRIGHT

            T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation)
            wide = wide.orient(current_orientation, target_orientation)

            '''
            Rotate IMG and Depth
            '''
            result["wide"]["image"] = rotate_tensor(result["wide"]["image"], current_orientation, target=target_orientation)
            if wide.has("depth"):
                result["wide"]["depth"] = rotate_tensor(result["wide"]["depth"], current_orientation, target=target_orientation)

            # No need for pose anymore.
            wide.RT = torch.eye(4)[None]
            wide.T_gravity = T_gravity[None]


            gt = PosedSensorInfo()        
            gt.RT = parse_transform_4x4_np(pose)[None]
            if depth_info is not None:
                gt.depth = depth_info

            sensor_info = SensorArrayInfo()
            sensor_info.wide = wide
            sensor_info.gt = gt

            result["meta"] = dict(video_id=video_id, timestamp=index)
            result["sensor_info"] = sensor_info

            # print("wide image",result['wide']['image'].shape)
            # print("meta",result['meta'])
            # print("image",wide.image.size) #image (1024, 768)

            index+=1
            yield result