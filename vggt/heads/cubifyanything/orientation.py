# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import numpy as np
import torch

from enum import Enum
from scipy.spatial.transform import Rotation

class ImageOrientation(Enum):
    UPRIGHT = 0
    LEFT = 1
    UPSIDE_DOWN = 2
    RIGHT = 3
    ORIGINAL = 4

ROT_Z = {
    (ImageOrientation.UPRIGHT, ImageOrientation.UPRIGHT): torch.tensor(Rotation.from_euler('z', 0).as_matrix()).float(),
    (ImageOrientation.LEFT, ImageOrientation.UPRIGHT): torch.tensor(Rotation.from_euler('z', np.pi / 2).as_matrix()).float(),
    (ImageOrientation.UPSIDE_DOWN, ImageOrientation.UPRIGHT): torch.tensor(Rotation.from_euler('z', np.pi).as_matrix()).float(),
    (ImageOrientation.RIGHT, ImageOrientation.UPRIGHT): torch.tensor(Rotation.from_euler('z', -np.pi / 2).as_matrix()).float(),

    # Inverses.
    (ImageOrientation.UPRIGHT, ImageOrientation.UPRIGHT): torch.tensor(Rotation.from_euler('z', 0).as_matrix()).float(),
    (ImageOrientation.UPRIGHT, ImageOrientation.LEFT): torch.tensor(Rotation.from_euler('z', -np.pi / 2).as_matrix()).float(),
    (ImageOrientation.UPRIGHT, ImageOrientation.UPSIDE_DOWN): torch.tensor(Rotation.from_euler('z', -np.pi).as_matrix()).float(),
    (ImageOrientation.UPRIGHT, ImageOrientation.RIGHT): torch.tensor(Rotation.from_euler('z', np.pi / 2).as_matrix()).float(),
}

ROT_K = {
    (ImageOrientation.UPRIGHT, ImageOrientation.UPRIGHT): 0,
    (ImageOrientation.LEFT, ImageOrientation.UPRIGHT): -1,
    (ImageOrientation.UPSIDE_DOWN, ImageOrientation.UPRIGHT): 2,
    (ImageOrientation.RIGHT, ImageOrientation.UPRIGHT): 1,

    # Inverses.
    (ImageOrientation.UPRIGHT, ImageOrientation.UPRIGHT): 0,
    (ImageOrientation.UPRIGHT, ImageOrientation.LEFT): 1,
    (ImageOrientation.UPRIGHT, ImageOrientation.UPSIDE_DOWN): -2,
    (ImageOrientation.UPRIGHT, ImageOrientation.RIGHT): -1
}

def get_orientation(pose):
    z_vec = pose[..., 2, :3]
    z_orien = torch.tensor(np.array(
        [
            [0.0, -1.0, 0.0],  # upright
            [-1.0, 0.0, 0.0],  # left
            [0.0, 1.0, 0.0],  # upside-down
            [1.0, 0.0, 0.0],
        ]  # right
    )).to(pose)

    corr = (z_orien @ z_vec.T).T
    corr_max = corr.argmax(dim=-1)

    return corr_max

def rotate_K(K, current, image_size, target=ImageOrientation.UPRIGHT):
    # TODO: use image_size to properly compute the new (cx, cy)
    if (current, target) in [(ImageOrientation.UPRIGHT, ImageOrientation.UPRIGHT)]:
        return K.clone()
    elif (current, target) in [(ImageOrientation.LEFT, ImageOrientation.UPRIGHT), (ImageOrientation.UPRIGHT, ImageOrientation.RIGHT)]:
        return torch.stack([
            torch.stack([K[:, 1, 1], K[:, 0, 1], K[:, 1, 2]], dim=1),
            torch.stack([K[:, 1, 0], K[:, 0, 0], K[:, 0, 2]], dim=1),
            torch.stack([K[:, 2, 0], K[:, 2, 1], K[:, 2, 2]], dim=1)
        ], dim=1).to(K)
    elif (current, target) in [(ImageOrientation.UPSIDE_DOWN, ImageOrientation.UPRIGHT), (ImageOrientation.UPRIGHT, ImageOrientation.UPSIDE_DOWN)]:
        return torch.stack([
            torch.stack([K[:, 0, 0], K[:, 0, 1], image_size[0] - K[:, 0, 2]], dim=1),
            torch.stack([K[:, 1, 0], K[:, 1, 1], image_size[1] - K[:, 1, 2]], dim=1),
            torch.stack([K[:, 2, 0], K[:, 2, 1], K[:, 2, 2]], dim=1)
        ], dim=1).to(K)
    elif (current, target) in [(ImageOrientation.RIGHT, ImageOrientation.UPRIGHT), (ImageOrientation.UPRIGHT, ImageOrientation.LEFT)]:
        return torch.stack([
            torch.stack([K[:, 1, 1], K[:, 0, 1], K[:, 1, 2]], dim=1),
            torch.stack([K[:, 1, 0], K[:, 0, 0], K[:, 0, 2]], dim=1),
            torch.stack([K[:, 2, 0], K[:, 2, 1], K[:, 2, 2]], dim=1)
        ], dim=1).to(K)

    raise ValueError("unknown orientation")

def rotate_pose(pose, current, target=ImageOrientation.UPRIGHT):
    rot_z = ROT_Z[(current, target)].to(pose)
    rot_z_4x4 = torch.eye(4, device=pose.device).float()
    rot_z_4x4[:3, :3] = rot_z

    return pose @ torch.linalg.inv(rot_z_4x4)

def rotate_xyz(xyz, current, target=ImageOrientation.UPRIGHT):
    rot_z = ROT_Z[(current, target)].to(xyz)
    return rot_z @ xyz

def rotate_tensor(tensor, current, target=ImageOrientation.UPRIGHT):
    return torch.rot90(tensor, ROT_K[(current, target)], dims=(-2, -1))
