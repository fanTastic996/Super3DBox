# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import math
import numpy as np
from PIL import Image
import PIL
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

from vggt.utils.geometry import closed_form_inverse_se3
import json
import torch
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union


#####################################################################################################################
def crop_image_depth_and_intrinsic_by_pp(
    image, depth_map, intrinsic, target_shape, track=None, filepath=None, strict=False
):
    """
    TODO: some names of width and height seem not consistent. Need to check.
    
    
    Crops the given image and depth map around the camera's principal point, as defined by `intrinsic`.
    Specifically:
      - Ensures that the crop is centered on (cx, cy).
      - Optionally pads the image (and depth map) if `strict=True` and the result is smaller than `target_shape`.
      - Shifts the camera intrinsic matrix (and `track` if provided) accordingly.

    Args:
        image (np.ndarray):
            Input image array of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array of shape (H, W), or None if not available.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3). The principal point is assumed to be at (intrinsic[1,2], intrinsic[0,2]).
        target_shape (tuple[int, int]):
            Desired output shape.
        track (np.ndarray or None):
            Optional array of shape (N, 2). Interpreted as (x, y) pixel coordinates. Will be shifted after cropping.
        filepath (str or None):
            An optional file path for debug logging (only used if strict mode triggers warnings).
        strict (bool):
            If True, will zero-pad to ensure the exact target_shape even if the cropped region is smaller.

    Raises:
        AssertionError:
            If the input image is smaller than `target_shape`.
        ValueError:
            If the cropped image is larger than `target_shape` (in strict mode), which should not normally happen.

    Returns:
        tuple:
            (cropped_image, cropped_depth_map, updated_intrinsic, updated_track)

            - cropped_image (np.ndarray): Cropped (and optionally padded) image.
            - cropped_depth_map (np.ndarray or None): Cropped (and optionally padded) depth map.
            - updated_intrinsic (np.ndarray): Intrinsic matrix adjusted for the crop.
            - updated_track (np.ndarray or None): Track array adjusted for the crop, or None if track was not provided.
    """
    original_size = np.array(image.shape)
    intrinsic = np.copy(intrinsic)

    if original_size[0] < target_shape[0]:
        error_message = (
            f"Width check failed: original width {original_size[0]} "
            f"is less than target width {target_shape[0]}."
        )
        print(error_message)
        raise AssertionError(error_message)

    if original_size[1] < target_shape[1]:
        error_message = (
            f"Height check failed: original height {original_size[1]} "
            f"is less than target height {target_shape[1]}."
        )
        print(error_message)
        raise AssertionError(error_message)

    # Identify principal point (cx, cy) from intrinsic
    cx = (intrinsic[1, 2])
    cy = (intrinsic[0, 2])

    # Compute how far we can crop in each direction
    if strict:
        half_x = min((target_shape[0] / 2), cx)
        half_y = min((target_shape[1] / 2), cy)
    else:
        half_x = min((target_shape[0] / 2), cx, original_size[0] - cx)
        half_y = min((target_shape[1] / 2), cy, original_size[1] - cy)

    # Compute starting indices
    start_x = math.floor(cx) - math.floor(half_x)
    start_y = math.floor(cy) - math.floor(half_y)

    assert start_x >= 0
    assert start_y >= 0

    # Compute ending indices
    if strict:
        end_x = start_x + target_shape[0]
        end_y = start_y + target_shape[1]
    else:
        end_x = start_x + 2 * math.floor(half_x)
        end_y = start_y + 2 * math.floor(half_y)

    # Perform the crop
    image = image[start_x:end_x, start_y:end_y, :]
    if depth_map is not None:
        depth_map = depth_map[start_x:end_x, start_y:end_y]

    # Shift the principal point in the intrinsic
    intrinsic[1, 2] = intrinsic[1, 2] - start_x
    intrinsic[0, 2] = intrinsic[0, 2] - start_y

    # Adjust track if provided
    if track is not None:
        track[:, 1] = track[:, 1] - start_x
        track[:, 0] = track[:, 0] - start_y

    # If strict, zero-pad if the new shape is smaller than target_shape
    if strict:
        if (image.shape[:2] != target_shape).any():
            print(f"{filepath} does not meet the target shape")
            current_h, current_w = image.shape[:2]
            target_h, target_w = target_shape[0], target_shape[1]
            pad_h = target_h - current_h
            pad_w = target_w - current_w
            if pad_h < 0 or pad_w < 0:
                raise ValueError(
                    f"The cropped image is bigger than the target shape: "
                    f"cropped=({current_h},{current_w}), "
                    f"target=({target_h},{target_w})."
                )
            image = np.pad(
                image,
                pad_width=((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if depth_map is not None:
                depth_map = np.pad(
                    depth_map,
                    pad_width=((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

    return image, depth_map, intrinsic, track


def resize_image_depth_and_intrinsic(
    image,
    depth_map,
    intrinsic,
    target_shape,
    original_size,
    track=None,
    pixel_center=True,
    safe_bound=4,
    rescale_aug=True,
):
    """
    Resizes the given image and depth map (if provided) to slightly larger than `target_shape`,
    updating the intrinsic matrix (and track array if present). Optionally uses random rescaling
    to create some additional margin (based on `rescale_aug`).

    Steps:
      1. Compute a scaling factor so that the resized result is at least `target_shape + safe_bound`.
      2. Apply an optional triangular random factor if `rescale_aug=True`.
      3. Resize the image with LANCZOS if downscaling, BICUBIC if upscaling.
      4. Resize the depth map with nearest-neighbor.
      5. Update the camera intrinsic and track coordinates (if any).

    Args:
        image (np.ndarray):
            Input image array (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array (H, W), or None if unavailable.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3).
        target_shape (np.ndarray or tuple[int, int]):
            Desired final shape (height, width).
        original_size (np.ndarray or tuple[int, int]):
            Original size of the image in (height, width).
        track (np.ndarray or None):
            Optional (N, 2) array of pixel coordinates. Will be scaled.
        pixel_center (bool):
            If True, accounts for 0.5 pixel center shift during resizing.
        safe_bound (int or float):
            Additional margin (in pixels) to add to target_shape before resizing.
        rescale_aug (bool):
            If True, randomly increase the `safe_bound` within a certain range to simulate augmentation.

    Returns:
        tuple:
            (resized_image, resized_depth_map, updated_intrinsic, updated_track)

            - resized_image (np.ndarray): The resized image.
            - resized_depth_map (np.ndarray or None): The resized depth map.
            - updated_intrinsic (np.ndarray): Camera intrinsic updated for new resolution.
            - updated_track (np.ndarray or None): Track array updated or None if not provided.

    Raises:
        AssertionError:
            If the shapes of the resized image and depth map do not match.
    """
    if rescale_aug:
        random_boundary = np.random.triangular(0, 0, 0.3)
        safe_bound = safe_bound + random_boundary * target_shape.max()

    resize_scales = (target_shape + safe_bound) / original_size
    max_resize_scale = np.max(resize_scales)
    intrinsic = np.copy(intrinsic)

    # Convert image to PIL for resizing
    image = Image.fromarray(image)
    input_resolution = np.array(image.size)
    output_resolution = np.floor(input_resolution * max_resize_scale).astype(int)
    image = image.resize(tuple(output_resolution), resample=lanczos if max_resize_scale < 1 else bicubic)
    image = np.array(image)

    if depth_map is not None:
        depth_map = cv2.resize(
            depth_map,
            output_resolution,
            fx=max_resize_scale,
            fy=max_resize_scale,
            interpolation=cv2.INTER_NEAREST,
        )

    actual_size = np.array(image.shape[:2])
    actual_resize_scale = np.max(actual_size / original_size)

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5

    intrinsic[:2, :] = intrinsic[:2, :] * actual_resize_scale

    if track is not None:
        track = track * actual_resize_scale

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5

    assert image.shape[:2] == depth_map.shape[:2]
    return image, depth_map, intrinsic, track


def threshold_depth_map(
    depth_map: np.ndarray,
    max_percentile: float = 99,
    min_percentile: float = 1,
    max_depth: float = -1,
) -> np.ndarray:
    """
    Thresholds a depth map using percentile-based limits and optional maximum depth clamping.

    Steps:
      1. If `max_depth > 0`, clamp all values above `max_depth` to zero.
      2. Compute `max_percentile` and `min_percentile` thresholds using nanpercentile.
      3. Zero out values above/below these thresholds, if thresholds are > 0.

    Args:
        depth_map (np.ndarray):
            Input depth map (H, W).
        max_percentile (float):
            Upper percentile (0-100). Values above this will be set to zero.
        min_percentile (float):
            Lower percentile (0-100). Values below this will be set to zero.
        max_depth (float):
            Absolute maximum depth. If > 0, any depth above this is set to zero.
            If <= 0, no maximum-depth clamp is applied.

    Returns:
        np.ndarray:
            Depth map (H, W) after thresholding. Some or all values may be zero.
            Returns None if depth_map is None.
    """
    if depth_map is None:
        return None

    depth_map = depth_map.astype(float, copy=True)

    # Optional clamp by max_depth
    if max_depth > 0:
        depth_map[depth_map > max_depth] = 0.0

    # Percentile-based thresholds
    depth_max_thres = (
        np.nanpercentile(depth_map, max_percentile) if max_percentile > 0 else None
    )
    depth_min_thres = (
        np.nanpercentile(depth_map, min_percentile) if min_percentile > 0 else None
    )

    # Apply the thresholds if they are > 0
    if depth_max_thres is not None and depth_max_thres > 0:
        depth_map[depth_map > depth_max_thres] = 0.0
    if depth_min_thres is not None and depth_min_thres > 0:
        depth_map[depth_map < depth_min_thres] = 0.0

    return depth_map


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    给定相机外参和内参，将深度图转换为世界坐标 (HxWx3)。
    返回世界坐标、中间的相机坐标以及有效深度的掩码。

    参数:
        depth_map (np.ndarray):
            形状为 (H, W) 的深度图。
        extrinsic (np.ndarray):
            形状为 (3, 4) 的外参矩阵，表示OpenCV约定中的相机位姿 (camera-from-world)。
        intrinsic (np.ndarray):
            形状为 (3, 3) 的内参矩阵。
        eps (float):
            用于阈值化有效深度的小 epsilon 值。

    返回:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (world_coords_points, cam_coords_points, point_mask)

            - world_coords_points: (H, W, 3) 的世界坐标系中的3D点数组。
            - cam_coords_points: (H, W, 3) 的相机坐标系中的3D点数组。
            - point_mask: (H, W) 的布尔数组，其中 True 表示有效的（非零）深度。
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # The extrinsic is camera-from-world, so invert it to transform camera->world
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = (
        np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world
    ) # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(
    depth_map: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """
    Unprojects a depth map into camera coordinates, returning (H, W, 3).

    Args:
        depth_map (np.ndarray):
            Depth map of shape (H, W).
        intrinsic (np.ndarray):
            3x3 camera intrinsic matrix.
            Assumes zero skew and standard OpenCV layout:
            [ fx   0   cx ]
            [  0  fy   cy ]
            [  0   0    1 ]

    Returns:
        np.ndarray:
            An (H, W, 3) array, where each pixel is mapped to (x, y, z) in the camera frame.
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert (
        intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0
    ), "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


def rotate_90_degrees(
    image, depth_map, extri_opencv, intri_opencv, clockwise=True, track=None
):
    """
    Rotates the input image, depth map, and camera parameters by 90 degrees.

    Applies one of two 90-degree rotations:
    - Clockwise
    - Counterclockwise (if clockwise=False)

    The extrinsic and intrinsic matrices are adjusted accordingly to maintain
    correct camera geometry. Track coordinates are also updated if provided.

    Args:
        image (np.ndarray):
            Input image of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map of shape (H, W), or None if not available.
        extri_opencv (np.ndarray):
            Extrinsic matrix (3x4) in OpenCV convention.
        intri_opencv (np.ndarray):
            Intrinsic matrix (3x3).
        clockwise (bool):
            If True, rotates the image 90 degrees clockwise; else 90 degrees counterclockwise.
        track (np.ndarray or None):
            Optional (N, 2) track array. Will be rotated accordingly.

    Returns:
        tuple:
            (
                rotated_image,
                rotated_depth_map,
                new_extri_opencv,
                new_intri_opencv,
                new_track
            )

            Where each is the updated version after the rotation.
    """
    image_height, image_width = image.shape[:2]

    # Rotate the image and depth map
    rotated_image, rotated_depth_map = rotate_image_and_depth_rot90(image, depth_map, clockwise)
    # Adjust the intrinsic matrix
    new_intri_opencv = adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise)

    if track is not None:
        new_track = adjust_track_rot90(track, image_width, image_height, clockwise)
    else:
        new_track = None

    # Adjust the extrinsic matrix
    new_extri_opencv = adjust_extrinsic_matrix_rot90(extri_opencv, clockwise)

    return (
        rotated_image,
        rotated_depth_map,
        new_extri_opencv,
        new_intri_opencv,
        new_track,
    )


def rotate_image_and_depth_rot90(image, depth_map, clockwise):
    """
    Rotates the given image and depth map by 90 degrees (clockwise or counterclockwise),
    using a transpose+flip pattern.

    Args:
        image (np.ndarray):
            Input image of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map of shape (H, W), or None if not available.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        tuple:
            (rotated_image, rotated_depth_map)
    """
    rotated_depth_map = None
    if clockwise:
        rotated_image = np.transpose(image, (1, 0, 2))  # Transpose height and width
        rotated_image = np.flip(rotated_image, axis=1)  # Flip horizontally
        if depth_map is not None:
            rotated_depth_map = np.transpose(depth_map, (1, 0))
            rotated_depth_map = np.flip(rotated_depth_map, axis=1)
    else:
        rotated_image = np.transpose(image, (1, 0, 2))  # Transpose height and width
        rotated_image = np.flip(rotated_image, axis=0)  # Flip vertically
        if depth_map is not None:
            rotated_depth_map = np.transpose(depth_map, (1, 0))
            rotated_depth_map = np.flip(rotated_depth_map, axis=0)
    return np.copy(rotated_image), np.copy(rotated_depth_map)


def adjust_extrinsic_matrix_rot90(extri_opencv, clockwise):
    """
    Adjusts the extrinsic matrix (3x4) for a 90-degree rotation of the image.

    The rotation is in the image plane. This modifies the camera orientation
    accordingly. The function applies either a clockwise or counterclockwise
    90-degree rotation.

    Args:
        extri_opencv (np.ndarray):
            Extrinsic matrix (3x4) in OpenCV convention.
        clockwise (bool):
            If True, rotate extrinsic for a 90-degree clockwise image rotation;
            otherwise, counterclockwise.

    Returns:
        np.ndarray:
            A new 3x4 extrinsic matrix after the rotation.
    """
    R = extri_opencv[:, :3]
    t = extri_opencv[:, 3]

    if clockwise:
        R_rotation = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])
    else:
        R_rotation = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])

    new_R = np.dot(R_rotation, R)
    new_t = np.dot(R_rotation, t)
    new_extri_opencv = np.hstack((new_R, new_t.reshape(-1, 1)))
    return new_extri_opencv


def adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise):
    """
    Adjusts the intrinsic matrix (3x3) for a 90-degree rotation of the image in the image plane.

    Args:
        intri_opencv (np.ndarray):
            Intrinsic matrix (3x3).
        image_width (int):
            Original width of the image.
        image_height (int):
            Original height of the image.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        np.ndarray:
            A new 3x3 intrinsic matrix after the rotation.
    """
    fx, fy, cx, cy = (
        intri_opencv[0, 0],
        intri_opencv[1, 1],
        intri_opencv[0, 2],
        intri_opencv[1, 2],
    )

    new_intri_opencv = np.eye(3)
    if clockwise:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = image_height - cy
        new_intri_opencv[1, 2] = cx
    else:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = cy
        new_intri_opencv[1, 2] = image_width - cx

    return new_intri_opencv


def adjust_track_rot90(track, image_width, image_height, clockwise):
    """
    Adjusts a track (N, 2) for a 90-degree rotation of the image in the image plane.

    Args:
        track (np.ndarray):
            (N, 2) array of pixel coordinates, each row is (x, y).
        image_width (int):
            Original image width.
        image_height (int):
            Original image height.
        clockwise (bool):
            Whether the rotation is 90 degrees clockwise or counterclockwise.

    Returns:
        np.ndarray:
            A new track of shape (N, 2) after rotation.
    """
    if clockwise:
        # (x, y) -> (y, image_width - 1 - x)
        new_track = np.stack((track[:, 1], image_width - 1 - track[:, 0]), axis=-1)
    else:
        # (x, y) -> (image_height - 1 - y, x)
        new_track = np.stack((image_height - 1 - track[:, 1], track[:, 0]), axis=-1)

    return new_track


def read_image_cv2(path: str, rgb: bool = True) -> np.ndarray:
    """
    Reads an image from disk using OpenCV, returning it as an RGB image array (H, W, 3).

    Args:
        path (str):
            File path to the image.
        rgb (bool):
            If True, convert the image to RGB.
            If False, leave the image in BGR/grayscale.

    Returns:
        np.ndarray or None:
            A numpy array of shape (H, W, 3) if successful,
            or None if the file does not exist or could not be read.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"File does not exist or is empty: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image={path}. Retrying...")
        img = cv2.imread(path)
        if img is None:
            print("Retry failed.")
            return None

    if rgb:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def read_depth(path: str, scale_adjustment=1.0) -> np.ndarray:
    """
    Reads a depth map from disk in either .exr or .png format. The .exr is loaded using OpenCV
    with the environment variable OPENCV_IO_ENABLE_OPENEXR=1. The .png is assumed to be a 16-bit
    PNG (converted from half float).

    Args:
        path (str):
            File path to the depth image. Must end with .exr or .png.
        scale_adjustment (float):
            A multiplier for adjusting the loaded depth values (default=1.0).

    Returns:
        np.ndarray:
            A float32 array (H, W) containing the loaded depth. Zeros or non-finite values
            may indicate invalid regions.

    Raises:
        ValueError:
            If the file extension is not supported.
    """
    if path.lower().endswith(".exr"):
        # Ensure OPENCV_IO_ENABLE_OPENEXR is set to "1"
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
        d[d > 1e9] = 0.0
    elif path.lower().endswith(".png"):
        # d = load_16big_png_depth(path)
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            print(f"Error: Failed to read depth image at {path}")
            raise ValueError(f"Failed to read depth image: {path}")
    else:
        raise ValueError(f'unsupported depth file name "{path}"')
    try:
        d = d * scale_adjustment
    except Exception as e:
        print(f"Error: {e}, path: {path}")
        return None
    
    
    # added by lyq
    # 替换NaN为0
    d[np.isnan(d)] = 0
    # 替换正无穷为float32最大值
    d[np.isposinf(d)] = np.finfo(np.float32).max
    # 替换负无穷为float32最小值
    d[np.isneginf(d)] = np.finfo(np.float32).min
    
    
    d[~np.isfinite(d)] = 0.0

    return d


def load_16big_png_depth(depth_png: str) -> np.ndarray:
    """
    Loads a 16-bit PNG as a half-float depth map (H, W), returning a float32 NumPy array.

    Implementation detail:
      - PIL loads 16-bit data as 32-bit "I" mode.
      - We reinterpret the bits as float16, then cast to float32.

    Args:
        depth_png (str):
            File path to the 16-bit PNG.

    Returns:
        np.ndarray:
            A float32 depth array of shape (H, W).
    """
    with Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth




# -----------------------------
# pose helpers
# -----------------------------
def _as_np_pose_4x4(pose: Any) -> np.ndarray:
    """Accept torch/np/list; (4,4) or (3,4) -> (4,4) np.float64"""
    if hasattr(pose, "detach"):  # torch
        T = pose.detach().cpu().numpy().astype(np.float64)
    else:
        T = np.asarray(pose, dtype=np.float64)

    if T.shape == (3, 4):
        T = np.vstack([T, np.array([0, 0, 0, 1], dtype=np.float64)])
    assert T.shape == (4, 4), f"pose must be (4,4) or (3,4), got {T.shape}"
    return T


def _invert_pose_4x4(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 pose."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_inv
    out[:3, 3] = t_inv
    return out


def _transform_points_c2w(points_cam: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """points_cam: (N,3), c2w: (4,4) -> points_world: (N,3)"""
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    N = pts.shape[0]
    homo = np.hstack([pts, np.ones((N, 1), dtype=np.float64)])  # (N,4)
    out = (c2w @ homo.T).T
    return out[:, :3]


# -----------------------------
# 2D convex hull (monotonic chain)
# -----------------------------
def _convex_hull_2d(points_xy: np.ndarray) -> np.ndarray:
    """points_xy: (N,2) -> hull (H,2) in CCW order"""
    pts = np.asarray(points_xy, dtype=np.float64)
    pts = np.unique(pts, axis=0)
    if pts.shape[0] <= 2:
        return pts

    # sort by x then y
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


# -----------------------------
# min-area bounding rectangle (scan hull edges)
# -----------------------------
def _min_area_rect(points_xy: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    points_xy: (N,2),建议输入 hull
    returns center_xy(2,), w, h, yaw(rad) (yaw: CCW from world +x)
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] == 0:
        raise ValueError("Empty point set.")
    if pts.shape[0] == 1:
        return pts[0], 0.0, 0.0, 0.0
    if pts.shape[0] == 2:
        d = pts[1] - pts[0]
        yaw = float(np.arctan2(d[1], d[0]))
        center = (pts[0] + pts[1]) * 0.5
        w = float(np.linalg.norm(d))
        h = 0.0
        return center, w, h, yaw

    best = None  # (area, center_xy, w, h, yaw)
    H = pts.shape[0]

    for i in range(H):
        p0 = pts[i]
        p1 = pts[(i + 1) % H]
        edge = p1 - p0
        ang = float(np.arctan2(edge[1], edge[0]))

        c, s = np.cos(-ang), np.sin(-ang)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)  # rotate by -ang
        pr = pts @ R.T

        minx, miny = pr.min(axis=0)
        maxx, maxy = pr.max(axis=0)
        w = maxx - minx
        h = maxy - miny
        area = w * h

        center_r = np.array([(minx + maxx) * 0.5, (miny + maxy) * 0.5], dtype=np.float64)

        c2, s2 = np.cos(ang), np.sin(ang)
        Rinv = np.array([[c2, -s2], [s2, c2]], dtype=np.float64)
        center_xy = center_r @ Rinv.T

        if best is None or area < best[0]:
            best = (area, center_xy, float(w), float(h), float(ang))

    _, center_xy, w, h, yaw = best
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return center_xy, w, h, yaw


def _rect_corners_xy(center_xy: np.ndarray, w: float, h: float, yaw: float) -> np.ndarray:
    """return 4 rectangle corners in xy, shape (4,2)"""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    hx, hy = w * 0.5, h * 0.5
    local = np.array([[ hx,  hy],
                      [ hx, -hy],
                      [-hx, -hy],
                      [-hx,  hy]], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ R.T + np.array([cx, cy], dtype=np.float64)[None, :]


def _merge_world_corners_from_points(world_points: np.ndarray, force_w_ge_h: bool = True) -> np.ndarray:
    """
    world_points: (K,3) : all corners from all frames (already in world)
    return merged world corners: (8,3)
    """
    zmin = float(world_points[:, 2].min())
    zmax = float(world_points[:, 2].max())

    hull = _convex_hull_2d(world_points[:, :2])
    center_xy, w, h, yaw = _min_area_rect(hull)

    # stabilize output (optional): always report w>=h
    if force_w_ge_h and h > w:
        w, h = h, w
        yaw = yaw + np.pi / 2.0
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

    c4 = _rect_corners_xy(center_xy, w, h, yaw)  # (4,2)
    bottom = np.hstack([c4, np.full((4, 1), zmin, dtype=np.float64)])
    top    = np.hstack([c4, np.full((4, 1), zmax, dtype=np.float64)])
    return np.vstack([bottom, top])  # (8,3)


def classify_shape(scale,
                   long_ratio=8.0,
                   flat_ratio=8.0,
                   round_ratio=1.8,
                   plane_ratio=2.5):
    s0, s1, s2 = np.sort(np.array(scale, dtype=np.float64))  # s0<=s1<=s2
    eps = 1e-9
    r21 = (s2 + eps) / (s1 + eps)  # longest / middle
    r10 = (s1 + eps) / (s0 + eps)  # middle / shortest

    # rod: one dimension much longer, cross-section similar
    if (r21 > long_ratio) and (r10 < round_ratio):
        return False

    # flat: thickness much smaller, in-plane dims not too skewed
    if (r10 > flat_ratio) and (r21 < plane_ratio):
        return False

    return True


@torch.no_grad()
def load_gt_corners_cam_multiframe(
    data_path: str,
    image_idx: Union[List[int], np.ndarray, torch.Tensor],
    json_name_fmt: str = "instances_{idx}.json",
    device: Union[str, torch.device, None] = None,
) -> List[torch.Tensor]:
    """
    For each idx in image_idx, read json and return corners in CAMERA coordinates.

    Returns:
        corners_cam_list: List[torch.FloatTensor], length F
            - corners_cam_list[i] has shape [Ni, 8, 3] on `device`
            - if a frame has no boxes, returns empty tensor with shape [0, 8, 3]
    Notes:
        Each json is a list of dicts, each dict must contain:
            b["corners"]  (8x3) in camera coordinates.
        If you also need ids, tell me and I can return (ids_list, corners_list).
    """
    # ---- normalize image_idx ----
    if isinstance(image_idx, torch.Tensor):
        idx_list = image_idx.detach().cpu().long().tolist()
    else:
        idx_list = list(np.asarray(image_idx).astype(int).tolist())
    F = len(idx_list)
    assert F > 0, "image_idx is empty."

    # ---- choose output device ----
    if device is None:
        device = "cpu"

    corners_cam_list: List[torch.Tensor] = []

    for idx in idx_list:
        json_path = os.path.join(data_path, json_name_fmt.format(idx=idx))
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing json: {json_path}")

        with open(json_path, "r") as f:
            inst_list = json.load(f)

        if not isinstance(inst_list, list):
            raise ValueError(f"Expect list in {json_path}, got {type(inst_list)}")

        corners_all = []
        for b in inst_list:
            corners_cam = np.asarray(b["corners"], dtype=np.float32).reshape(8, 3)
            corners_all.append(corners_cam)

        if len(corners_all) == 0:
            # no boxes in this frame
            corners_t = np.zeros((1, 8, 3), dtype=np.float32)
        else:
            corners_np = np.stack(corners_all, axis=0)  # [Ni,8,3]
            corners_t = np.array(corners_np, dtype=np.float32)

        corners_cam_list.append(corners_t)

    return corners_cam_list


# =====================================================================
# QCOD (Query-Coevolved Object Detection) GT loaders
# =====================================================================

@torch.no_grad()
def _qcod_sanity_check_via_per_frame(
    scene_path: str,
    scene_ids: List[str],
    scene_corners_cam1: np.ndarray,
    first_frame_idx: int,
    tol: float = 1.0,
) -> None:
    """
    Closed-loop verification that scene GT was correctly transformed into the
    sampled first frame's camera coordinate system.

    For each instance present in both scene instances.json and the per-frame
    instances/{first_frame_idx}.json, compare the transformed scene corners
    (scene_corners_cam1) against the per-frame JSON corners (which are natively
    in that frame's camera coords). If the average discrepancy exceeds tol,
    this almost certainly indicates a c2w/w2c convention mix-up upstream.

    Args:
        scene_path: directory containing instances/ subfolder
        scene_ids: List[str] UUIDs aligned with scene_corners_cam1 rows
        scene_corners_cam1: [N, 8, 3] scene GT after the world -> cam1 transform
        first_frame_idx: int, the SAMPLED first frame index (image_idxs[0])
                         NOT a hardcoded 0. CA1M sampling picks a random start.
        tol: meters. The residual between scene OBB (global fit) and per-frame
             OBB is typically 0.2-0.5m; anything > 1.0m means transform is wrong.
    """
    pf_path = os.path.join(scene_path, "instances", f"{first_frame_idx}.json")
    if not os.path.exists(pf_path):
        return

    try:
        with open(pf_path, "r") as f:
            pf_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return

    pf_by_id: Dict[str, np.ndarray] = {}
    for inst in pf_data:
        try:
            pf_by_id[inst["id"]] = np.asarray(
                inst["corners"], dtype=np.float32
            ).reshape(8, 3)
        except (KeyError, ValueError):
            continue

    errors: List[float] = []
    for i, iid in enumerate(scene_ids):
        if iid not in pf_by_id:
            continue
        pred = scene_corners_cam1[i].astype(np.float64)  # [8, 3]
        gt = pf_by_id[iid].astype(np.float64)            # [8, 3]
        # Chamfer-ish: average nearest-neighbor distance in both directions
        diff = pred[:, None, :] - gt[None, :, :]         # [8, 8, 3]
        dist = np.linalg.norm(diff, axis=-1)             # [8, 8]
        err = 0.5 * (dist.min(axis=1).mean() + dist.min(axis=0).mean())
        errors.append(float(err))

    if not errors:
        return

    mean_err = float(np.mean(errors))
    if mean_err >= tol:
        raise AssertionError(
            f"[QCOD sanity] Scene GT coord transform failed! "
            f"scene={scene_path}, first_frame_idx={first_frame_idx}, "
            f"mean_err={mean_err:.3f}m (tol={tol}m). "
            f"Likely c2w/w2c convention mismatch — check first_frame_w2c."
        )


@torch.no_grad()
def load_scene_gt_in_first_frame(
    scene_path: str,
    first_frame_w2c: np.ndarray,
    first_frame_idx_for_check: int = None,
    sanity_check: bool = True,
    ignored_categories: Tuple[str, ...] = ("wall", "floor", "ceiling"),
) -> Dict[str, Any]:
    """
    Load scene-level GT from ``{scene_path}/instances.json`` and transform it
    from the raw world coordinate system into the SAMPLED first frame's
    camera coordinate system.

    IMPORTANT CONVENTIONS (do not guess, verified via closed-loop test):
      * ``instances.json`` ``corners`` / ``R`` / ``position`` are in the raw
        dataset **world** coordinate system (NOT already in any camera frame).
      * ``scale`` is the **full** box extent along each local axis — the
        caller will see ``scale / 2`` (half-extent) in the returned dict.
      * ``first_frame_w2c`` must be **w2c** (world-to-camera). In CA1M the
        raw ``all_poses.npy`` stores c2w, so the caller must pass either
        ``closed_form_inverse_se3(all_poses[idx])`` or the already-inverted
        ``seq_poses[idx]`` from ``ca1m.py`` (see line 225).
      * ``first_frame_idx_for_check`` must be the **sampled** frame index
        ``image_idxs[0]``, not a hardcoded 0.

    Args:
        scene_path: scene directory (contains ``instances.json`` and
            ``instances/{idx}.json`` files).
        first_frame_w2c: [4, 4] world-to-camera matrix for the sampled first
            frame (OpenCV convention).
        first_frame_idx_for_check: int, the sampled first frame index. Only
            used by the optional sanity check; can be None to disable.
        sanity_check: if True and first_frame_idx_for_check is not None,
            run the closed-loop check after transforming.
        ignored_categories: categories filtered out (walls/floors/ceilings
            are not meaningful as target instances).

    Returns:
        Dict with:
            ids:      List[str], length N (instance UUIDs)
            corners:  np.ndarray [N, 8, 3] float32 — sampled first frame camera
                      system (NOT yet normalized by avg_scale)
            R:        np.ndarray [N, 3, 3] float32 — rotation in cam1 frame
            scale:    np.ndarray [N, 3]    float32 — HALF extents (scale / 2)
            category: List[str], length N
    """
    instances_path = os.path.join(scene_path, "instances.json")
    if not os.path.exists(instances_path):
        return {
            "ids": [],
            "corners": np.zeros((0, 8, 3), dtype=np.float32),
            "R": np.zeros((0, 3, 3), dtype=np.float32),
            "scale": np.zeros((0, 3), dtype=np.float32),
            "category": [],
        }

    with open(instances_path, "r") as f:
        scene_data = json.load(f)

    ids: List[str] = []
    corners_list: List[np.ndarray] = []
    R_list: List[np.ndarray] = []
    scale_list: List[np.ndarray] = []
    cat_list: List[str] = []

    for inst in scene_data:
        cat = inst.get("category", "")
        if cat in ignored_categories:
            continue
        if "id" not in inst or "corners" not in inst:
            continue
        try:
            c = np.asarray(inst["corners"], dtype=np.float32).reshape(8, 3)
        except (ValueError, TypeError):
            continue
        # R and scale are not strictly required for the pipeline (derived
        # supervision only uses corners), but the QCOD loss needs them for
        # rotation regression and optional size loss. Default to identity/zeros
        # if missing to keep the loader robust.
        try:
            R = np.asarray(
                inst.get("R", np.eye(3, dtype=np.float32)), dtype=np.float32
            ).reshape(3, 3)
        except (ValueError, TypeError):
            R = np.eye(3, dtype=np.float32)
        try:
            s = np.asarray(
                inst.get("scale", np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(3)
        except (ValueError, TypeError):
            s = np.zeros(3, dtype=np.float32)

        ids.append(inst["id"])
        corners_list.append(c)
        R_list.append(R)
        scale_list.append(s)
        cat_list.append(cat)

    if len(corners_list) == 0:
        return {
            "ids": [],
            "corners": np.zeros((0, 8, 3), dtype=np.float32),
            "R": np.zeros((0, 3, 3), dtype=np.float32),
            "scale": np.zeros((0, 3), dtype=np.float32),
            "category": [],
        }

    corners_world = np.stack(corners_list, axis=0)  # [N, 8, 3]
    R_world = np.stack(R_list, axis=0)              # [N, 3, 3]
    scale_full = np.stack(scale_list, axis=0)       # [N, 3] (full extent)

    # --- world -> sampled first frame camera ---
    # X_cam1 = R_w2c @ X_world + t_w2c
    w2c = np.asarray(first_frame_w2c, dtype=np.float32)
    R_w2c = w2c[:3, :3]                             # [3, 3]
    t_w2c = w2c[:3, 3]                              # [3]
    corners_cam1 = corners_world @ R_w2c.T + t_w2c[None, None, :]  # [N, 8, 3]
    R_cam1 = R_w2c[None] @ R_world                                 # [N, 3, 3]

    # JSON scale is FULL extent; downstream pipeline uses HALF extent.
    # This matches Direct3DBoxHead.size = exp(log_size) (half-extent) convention.
    half_extent = scale_full * 0.5                                 # [N, 3]

    result = {
        "ids": ids,
        "corners": corners_cam1.astype(np.float32),
        "R": R_cam1.astype(np.float32),
        "scale": half_extent.astype(np.float32),
        "category": cat_list,
    }

    if sanity_check and first_frame_idx_for_check is not None:
        _qcod_sanity_check_via_per_frame(
            scene_path=scene_path,
            scene_ids=ids,
            scene_corners_cam1=result["corners"],
            first_frame_idx=int(first_frame_idx_for_check),
            tol=1.0,
        )

    return result


# -----------------------------
# main function
# -----------------------------
@torch.no_grad()
def merge_scene_gt_corners_world_multiframe(
    data_path: str,
    image_idx: Union[List[int], np.ndarray, torch.Tensor],
    extrinsic: Union[np.ndarray, torch.Tensor],  # [F,4,4] (or [F,3,4])
    json_name_fmt: str = "instances_{idx}.json",
    extrinsic_is_w2c: bool = True,     # True: w2c -> invert to c2w; False: already c2w
    keep_single_view: bool = True,     # id only appears once -> directly transform corners
    force_w_ge_h: bool = True,         # stabilize w/h & yaw
    device: Union[str, torch.device, None] = None,
) -> torch.Tensor:
    """
    Returns:
        corners_world: torch.FloatTensor [N,8,3]  (scene-level per id, in world)

    Notes:
      - Each json is a list of dicts, each dict must contain:
          b["id"] (string) and b["corners"] (8x3) in camera coordinates.
      - Merge rule:
          xy: all world corners -> 2D hull -> min-area rectangle
          z : union (min/max)
    """
    # ---- normalize image_idx ----
    if isinstance(image_idx, torch.Tensor):
        idx_list = image_idx.detach().cpu().long().tolist()
    else:
        idx_list = list(np.asarray(image_idx).astype(int).tolist())
    F = len(idx_list)
    assert F > 0, "image_idx is empty."

    # ---- normalize extrinsic to [F,4,4] numpy ----
    if hasattr(extrinsic, "detach"):  # torch
        ext_np = extrinsic.detach().cpu().numpy()
    else:
        ext_np = np.asarray(extrinsic)
    assert ext_np.shape[0] == F, f"extrinsic first dim must match len(image_idx): {ext_np.shape[0]} vs {F}"

    c2w_by_frame: Dict[int, np.ndarray] = {}
    for i, idx in enumerate(idx_list):
        Ti = _as_np_pose_4x4(ext_np[i])
        if extrinsic_is_w2c:
            Ti = _invert_pose_4x4(Ti)  # w2c -> c2w
        c2w_by_frame[idx] = Ti

    # ---- choose output device ----
    if device is None:
        device = extrinsic.device if hasattr(extrinsic, "device") else "cpu"

    # ---- read all frames + group by id (insertion order) ----
    buckets: "OrderedDict[str, List[Tuple[int, np.ndarray]]]" = OrderedDict()
    for idx in idx_list:
        json_path = os.path.join(data_path, json_name_fmt.format(idx=idx))
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing json: {json_path}")

        with open(json_path, "r") as f:
            inst_list = json.load(f)

        if not isinstance(inst_list, list):
            raise ValueError(f"Expect list in {json_path}, got {type(inst_list)}")

        for b in inst_list:
            bid = b["id"]
            category = b["category"]
            if category in ['wall','floor','ceiling']: #TODO:added by lyq to ignore wall, floor, ceiling 26-1-19
                continue
            # scale = b["scale"]
            # if not classify_shape(scale): #TODO:added by lyq to ignore extremely elongated or thin planar objects 26-1-19
            #     continue
            
            corners_cam = np.asarray(b["corners"], dtype=np.float64).reshape(8, 3)
            if bid not in buckets:
                buckets[bid] = []
            buckets[bid].append((idx, corners_cam))

    # ---- merge per id ----
    merged_list = []
    for bid, items in buckets.items():
        if keep_single_view and len(items) == 1:
            fidx, corners_cam = items[0]
            corners_w = _transform_points_c2w(corners_cam, c2w_by_frame[fidx])  # (8,3)
        else:
            all_w = []
            for fidx, corners_cam in items:
                all_w.append(_transform_points_c2w(corners_cam, c2w_by_frame[fidx]))
            all_w = np.concatenate(all_w, axis=0)  # (8*K,3)
            corners_w = _merge_world_corners_from_points(all_w, force_w_ge_h=force_w_ge_h)  # (8,3)

        merged_list.append(corners_w)
    # if no valid GT boxes found, return empty tensor
    if len(merged_list) == 0:
        corners_world = np.zeros((1, 8, 3), dtype=np.float32)
        return corners_world
    
    corners_world = np.stack(merged_list, axis=0)  # [N,8,3]
    return corners_world


def filter_gt_boxes_by_2d_valid_area_ratio_np(
    filtered_bbox_corners,   # np.ndarray [N, 8, 3] (world coords)
    intrinsics,              # np.ndarray [S, 3, 3]
    extrinsics,              # np.ndarray [S, 4, 4]  (w2c by default; see extrinsic_is_c2w)
    H, W,
    thr=0.20,
    extrinsic_is_c2w=False,  # True if extrinsics is c2w, will invert to w2c
    eps=1e-6,
    return_debug=False,
):
    """
    Numpy版：根据 3D GT box 在多视角投影后的 2D bbox “有效区域比例”过滤。
    - 投影得到每个 view 的 2D bbox (xyxy)，允许负数/超出边界（不截断）
    - ratio = bbox与图像边界相交面积 / bbox自身面积
    - keep_if_any_view 固定为 True：只要任意一张图 ratio>=thr 就保留

    注意：
    - 只使用 z>eps 的角点参与 min/max；某 view 没有任何角点在前方 -> ratio=0
    - bbox 面积用原始 bbox（不clip），intersection 用 clip 后计算
    """

    corners = np.asarray(filtered_bbox_corners)
    K = np.asarray(intrinsics)
    E = np.asarray(extrinsics)

    assert corners.ndim == 3 and corners.shape[1:] == (8, 3), corners.shape
    assert K.ndim == 3 and K.shape[1:] == (3, 3), K.shape
    assert E.ndim == 3 and E.shape[1:] == (4, 4), E.shape

    N = corners.shape[0]
    S = K.shape[0]
    assert E.shape[0] == S

    # use float64 for robustness
    corners = corners.astype(np.float64, copy=False)
    K = K.astype(np.float64, copy=False)
    E = E.astype(np.float64, copy=False)

    # w2c
    if extrinsic_is_c2w:
        w2c = np.linalg.inv(E)
    else:
        w2c = E

    # corners_h: [N,8,4]
    ones = np.ones((N, 8, 1), dtype=np.float64)
    corners_h = np.concatenate([corners, ones], axis=-1)

    # Expand to views: [S,N,8,4]
    corners_h = np.broadcast_to(corners_h[None, ...], (S, N, 8, 4))

    # IMPORTANT FIX: per-view transpose, NOT w2c.T
    w2c_T = np.transpose(w2c, (0, 2, 1))  # [S,4,4]

    # cam_h = corners_h @ w2c^T  (row vectors)
    cam_h = np.einsum("snkj,sji->snki", corners_h, w2c_T)  # [S,N,8,4]
    cam = cam_h[..., :3]                                   # [S,N,8,3]
    x = cam[..., 0]
    y = cam[..., 1]
    z = cam[..., 2]

    front = z > eps                                        # [S,N,8]
    any_front = np.any(front, axis=-1)                     # [S,N]

    # intrinsics params
    fx = K[:, 0, 0].reshape(S, 1, 1)
    fy = K[:, 1, 1].reshape(S, 1, 1)
    cx = K[:, 0, 2].reshape(S, 1, 1)
    cy = K[:, 1, 2].reshape(S, 1, 1)

    z_safe = np.where(front, z, 1.0)                       # avoid div0
    u = fx * (x / z_safe) + cx                              # [S,N,8]
    v = fy * (y / z_safe) + cy

    # ignore invalid points in min/max
    u_min_src = np.where(front, u,  np.inf)
    v_min_src = np.where(front, v,  np.inf)
    u_max_src = np.where(front, u, -np.inf)
    v_max_src = np.where(front, v, -np.inf)

    x1 = np.min(u_min_src, axis=-1)                         # [S,N]
    y1 = np.min(v_min_src, axis=-1)
    x2 = np.max(u_max_src, axis=-1)
    y2 = np.max(v_max_src, axis=-1)

    # bbox area (no clip)
    bw = np.clip(x2 - x1, 0.0, None)
    bh = np.clip(y2 - y1, 0.0, None)
    area = bw * bh                                          # [S,N]

    # intersection with image bounds
    ix1 = np.clip(x1, 0.0, float(W))
    iy1 = np.clip(y1, 0.0, float(H))
    ix2 = np.clip(x2, 0.0, float(W))
    iy2 = np.clip(y2, 0.0, float(H))
    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih                                         # [S,N]

    ratio = np.zeros_like(inter, dtype=np.float64)
    valid = (area > 0.0) & any_front
    ratio[valid] = inter[valid] / (area[valid] + 1e-12)

    # keep_if_any_view=True
    score = np.max(ratio, axis=0)                            # [N]
    keep = score >= float(thr)
    kept_corners = corners[keep]                             # [N_keep,8,3]

    if not return_debug:
        return kept_corners, keep

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)          # [S,N,4]
    return kept_corners, keep, boxes_xyxy, ratio
