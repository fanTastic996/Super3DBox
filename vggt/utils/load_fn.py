# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import os
from typing import Dict, Any, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords

def load_and_preprocess_images_original(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def load_and_preprocess_images_ca1m(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes

    for i, image_path in enumerate(image_path_list):
        # Open image
        img = Image.open(image_path)
        
        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
    
    
    
    # Convert to numpy and visualize
    # images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to [N, H, W, 3]
    # # Create output directory if it doesn't exist
    # output_dir = "/home/lanyuqing/myproject/vggt/vis_results"
    # os.makedirs(output_dir, exist_ok=True)
    # # Save each image
    # for i, img_np in enumerate(images_np):
    #     # Clip values to [0, 1] and convert to uint8
    #     img_np = np.clip(img_np, 0, 1)
    #     img_np = (img_np * 255).astype(np.uint8)
        
    #     # Convert to PIL Image and save
    #     img_pil = Image.fromarray(img_np)
    #     output_path = os.path.join(output_dir, f"processed_image_{i}.png")
    #     img_pil.save(output_path)
    #     print(f"Saved visualized image to: {output_path}")
    
    return images #[N, 3, H=518, W=518]


def load_and_preprocess_images_resize(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes

    for i, image_path in enumerate(image_path_list):
        # Open image
        img = Image.open(image_path)
        
        #TODO: added 26-1-18 [1024, 768] -> [512, 384]
        # img = img.resize((img.size[0]//2, img.size[1]//2), Image.Resampling.BICUBIC)
        
        
        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        
        # Original behavior: set width to 518px
        #TODO:这里需要修改，因为有些图是长边是518，有些是短边是518
        if height >= width:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop":
            if new_height > target_size:
                start_y = (new_height - target_size) // 2
                img = img[:, start_y : start_y + target_size, :]
            if new_width > target_size:
                start_x = (new_width - target_size) // 2
                img = img[:, :, start_x : start_x + target_size]

        

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
    
    
    
    # Convert to numpy and visualize
    # images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to [N, H, W, 3]
    # # Create output directory if it doesn't exist
    # output_dir = "/home/lanyuqing/myproject/vggt/vis_results"
    # os.makedirs(output_dir, exist_ok=True)
    # # Save each image
    # for i, img_np in enumerate(images_np):
    #     # Clip values to [0, 1] and convert to uint8
    #     img_np = np.clip(img_np, 0, 1)
    #     img_np = (img_np * 255).astype(np.uint8)
        
    #     # Convert to PIL Image and save
    #     img_pil = Image.fromarray(img_np)
    #     output_path = os.path.join(output_dir, f"processed_image_{i}.png")
    #     img_pil.save(output_path)
    #     print(f"Saved visualized image to: {output_path}")
    
    return images #[N, 3, H=518, W=518]


def load_and_preprocess_images_mine(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes

    for i, image_path in enumerate(image_path_list):
        # Open image
        img = Image.open(image_path)
        
        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size


        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            #TODO:这里需要修改，因为有些图是长边是518，有些是短边是518
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        
        

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        # new_width = width
        # new_height = height
        
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop":
            # Crop both height and width to target_size, centered
            start_y = max((new_height - target_size) // 2, 0)
            start_x = max((new_width - target_size) // 2, 0)
            img = img[:, start_y : start_y + target_size, start_x : start_x + target_size]
            

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images
    images = torch.stack(images)  # concatenate images
    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
    
    
    
    # Convert to numpy and visualize
    # images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to [N, H, W, 3]
    # # Create output directory if it doesn't exist
    # output_dir = "/home/lanyuqing/myproject/vggt/vis_results"
    # os.makedirs(output_dir, exist_ok=True)
    # # Save each image
    # for i, img_np in enumerate(images_np):
    #     # Clip values to [0, 1] and convert to uint8
    #     img_np = np.clip(img_np, 0, 1)
    #     img_np = (img_np * 255).astype(np.uint8)
        
    #     # Convert to PIL Image and save
    #     img_pil = Image.fromarray(img_np)
    #     output_path = os.path.join(output_dir, f"processed_image_{i}.png")
    #     img_pil.save(output_path)
    #     print(f"Saved visualized image to: {output_path}")
    
    return images #[N, 3, H=518, W=518]

def save_colored_pointcloud(points_xyz: np.ndarray, colors_rgb: np.ndarray, out_path: str):
    """
    points_xyz: (N,3) float
    colors_rgb: (N,3) float in [0,1] OR uint8 in [0,255]
    out_path: end with .ply or .pcd (recommended .ply)
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3, points_xyz.shape
    assert colors_rgb.ndim == 2 and colors_rgb.shape[1] == 3, colors_rgb.shape
    assert points_xyz.shape[0] == colors_rgb.shape[0], (points_xyz.shape, colors_rgb.shape)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --- normalize colors to uint8 [0,255] for PLY writing ---
    if colors_rgb.dtype != np.uint8:
        c = colors_rgb.astype(np.float32)
        # if looks like [0,1], scale; otherwise assume already [0,255]
        if c.max() <= 1.0 + 1e-6:
            c = np.clip(c, 0.0, 1.0) * 255.0
        c = np.clip(c, 0.0, 255.0).astype(np.uint8)
    else:
        c = colors_rgb

    # Try Open3D first
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))

        # Open3D expects float colors in [0,1]
        pcd.colors = o3d.utility.Vector3dVector((c.astype(np.float32) / 255.0))

        ok = o3d.io.write_point_cloud(out_path, pcd, write_ascii=False, compressed=False)
        if not ok:
            raise RuntimeError(f"Open3D failed to write point cloud: {out_path}")
        print(f"[OK] Saved point cloud with {len(points_xyz)} points to: {out_path}")
        return

    except ImportError:
        print("[WARN] open3d not installed, fallback to manual PLY writer...")

    # --- Fallback: manual binary little-endian PLY writer (supports .ply only) ---
    if not out_path.lower().endswith(".ply"):
        raise ValueError("Fallback writer only supports .ply. Please install open3d for .pcd/.ply support.")

    pts = points_xyz.astype(np.float32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    with open(out_path, "wb") as f:
        f.write(header)
        # interleave xyz + rgb
        data = np.empty(pts.shape[0], dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                                            ("r","u1"),("g","u1"),("b","u1")])
        data["x"], data["y"], data["z"] = pts[:,0], pts[:,1], pts[:,2]
        data["r"], data["g"], data["b"] = c[:,0], c[:,1], c[:,2]
        f.write(data.tobytes())

    print(f"[OK] Saved point cloud with {len(points_xyz)} points to: {out_path}")

def  save_gt_data(gt_world_points, valid_mask, depths, save_dir):
    """Save ground truth data for comparison."""
    # if gt_data is None:
    #     return

    gt_dir = save_dir #os.path.join(save_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    # Save GT depth visualizations
    save_depth_visualizations(depths, gt_dir)

    # Save GT data arrays
    np.savez(
        os.path.join(gt_dir, "gt_data.npz"),
        gt_world_points=gt_world_points,
        valid_mask=valid_mask,
    )

def save_depth_visualizations(depth_maps: np.ndarray, save_dir: str):
    """
    Save depth map visualizations with proper normalization and multiple visualization modes.

    Args:
        depth_maps: Array of depth maps with shape (N, H, W)
        save_dir: Directory to save visualizations
    """
    output_dir = os.path.join(save_dir, "pred_depths")
    os.makedirs(output_dir, exist_ok=True)

    # Compute depth statistics
    valid_depths = depth_maps[depth_maps > 0]  # Filter out invalid depths
    valid_depths = valid_depths / 1000.0

    depth_min = np.percentile(valid_depths, 1)   # Use percentiles to handle outliers
    depth_max = np.percentile(valid_depths, 99)
    depth_mean = np.mean(valid_depths)
    depth_std = np.std(valid_depths)

    print(f"Depth statistics - Min: {depth_min:.3f}, Max: {depth_max:.3f}, "
                f"Mean: {depth_mean:.3f}, Std: {depth_std:.3f}")

    # Save depth statistics
    stats = {
        'min': depth_min,
        'max': depth_max,
        'mean': depth_mean,
        'std': depth_std,
        'percentile_1': depth_min,
        'percentile_99': depth_max,
        'valid_pixel_ratio': len(valid_depths) / depth_maps.size
    }
    np.save(os.path.join(output_dir, 'depth_statistics.npy'), stats)

    # Visualization modes
    vis_modes = {
        'jet': cv2.COLORMAP_JET,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'turbo': cv2.COLORMAP_TURBO
    }

    images_dict = {mode: [] for mode in vis_modes.keys()}

    for i, depth_map in enumerate(depth_maps):
        # Handle invalid depths (set to min depth for visualization)
        depth_vis = depth_map.copy()
        depth_vis[depth_vis <= 0] = depth_min

        # Normalize depth to [0, 1] using robust statistics
        depth_normalized = np.clip((depth_vis - depth_min) / (depth_max - depth_min), 0, 1)

        # Convert to uint8 for visualization
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # Save raw normalized depth
        np.save(os.path.join(output_dir, f'depth_normalized_{i:04d}.npy'), depth_normalized)

        # Generate visualizations with different colormaps
        for mode_name, colormap in vis_modes.items():
            # Apply colormap
            depth_colored = cv2.applyColorMap(depth_uint8, colormap)

            # Create mode-specific directory
            mode_dir = os.path.join(output_dir, mode_name)
            os.makedirs(mode_dir, exist_ok=True)

            # Save individual frame
            img_path = os.path.join(mode_dir, f'frame_{i:04d}.png')
            cv2.imwrite(img_path, depth_colored)
            images_dict[mode_name].append(Image.open(img_path))

            # Save with depth scale bar (for the first mode only)
            if mode_name == 'jet':
                add_depth_scale_bar(depth_colored, depth_min, depth_max, img_path.replace('.png', '_with_scale.png'))

    # Create animated GIFs for each visualization mode
    for mode_name, images in images_dict.items():
        if images:
            gif_path = os.path.join(output_dir, f'depth_maps_{mode_name}.gif')
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=200,  # Slower animation for better viewing
                loop=0
            )

    # Create a comparison visualization (side by side different colormaps)
    create_comparison_visualization(depth_maps, depth_min, depth_max, output_dir)
    

def add_depth_scale_bar(depth_image: np.ndarray, depth_min: float, depth_max: float, save_path: str):
    """Add a depth scale bar to the visualization."""


    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Display the depth image
    ax.imshow(cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    # Add scale bar
    scale_bar_height = 20
    scale_bar_width = 200
    scale_bar_x = depth_image.shape[1] - scale_bar_width - 20
    scale_bar_y = depth_image.shape[0] - scale_bar_height - 40

    # Create gradient for scale bar
    gradient = np.linspace(0, 1, scale_bar_width).reshape(1, -1)
    gradient = np.repeat(gradient, scale_bar_height, axis=0)

    # Apply same colormap as depth image
    scale_colored = cv2.applyColorMap((gradient * 255).astype(np.uint8), cv2.COLORMAP_JET)
    scale_colored = cv2.cvtColor(scale_colored, cv2.COLOR_BGR2RGB)

    # Add scale bar to image
    ax.imshow(scale_colored, extent=[scale_bar_x, scale_bar_x + scale_bar_width,
                                    scale_bar_y + scale_bar_height, scale_bar_y])

    # Add text labels
    ax.text(scale_bar_x, scale_bar_y - 5, f'{depth_min:.2f}m',
            color='white', fontsize=10, ha='left', weight='bold')
    ax.text(scale_bar_x + scale_bar_width, scale_bar_y - 5, f'{depth_max:.2f}m',
            color='white', fontsize=10, ha='right', weight='bold')
    ax.text(scale_bar_x + scale_bar_width//2, scale_bar_y - 5, 'Depth',
            color='white', fontsize=10, ha='center', weight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def create_comparison_visualization(depth_maps: np.ndarray, depth_min: float, depth_max: float, output_dir: str):
    """Create a comparison visualization showing different colormaps side by side."""

    # Select a representative frame (middle frame)
    mid_idx = len(depth_maps) // 2
    depth_map = depth_maps[mid_idx]

    # Normalize depth
    depth_vis = depth_map.copy()
    depth_vis[depth_vis <= 0] = depth_min
    depth_normalized = np.clip((depth_vis - depth_min) / (depth_max - depth_min), 0, 1)

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colormaps = ['jet', 'viridis', 'plasma', 'turbo']
    cv_colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA, cv2.COLORMAP_TURBO]

    for i, (cmap_name, cv_cmap) in enumerate(zip(colormaps, cv_colormaps)):
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv_cmap)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        axes[i].imshow(depth_colored)
        axes[i].set_title(f'{cmap_name.capitalize()} Colormap', fontsize=12, weight='bold')
        axes[i].axis('off')

    plt.suptitle(f'Depth Visualization Comparison (Frame {mid_idx})\n'
                f'Depth Range: {depth_min:.2f}m - {depth_max:.2f}m',
                fontsize=14, weight='bold')
    plt.tight_layout()

    comparison_path = os.path.join(output_dir, 'colormap_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

def _infer_depth_path_from_rgb(
    rgb_path: str,
    depth_dir: Optional[str] = None,
    rgb_dir_token: str = "rgb",
    depth_dir_token: str = "depth",
    exts: Tuple[str, ...] = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".tiff", ".TIFF"),
) -> str:
    """
    Try to infer depth path from an RGB path:
      - if depth_dir is given: depth_path = depth_dir / basename(rgb)
      - else: replace a directory token 'rgb' -> 'depth' in the path
    It keeps the same filename (including extension). If not found, it will try common depth extensions.
    """
    base = os.path.basename(rgb_path)
    if depth_dir is not None:
        cand = os.path.join(depth_dir, base)
        if os.path.exists(cand):
            return cand
        # try changing extension
        stem, _ = os.path.splitext(base)
        for e in exts:
            cand2 = os.path.join(depth_dir, stem + e)
            if os.path.exists(cand2):
                return cand2
        return cand  # fall back (will raise later)

    # replace token in full path
    parts = rgb_path.split(os.sep)
    parts2 = [depth_dir_token if p == rgb_dir_token else p for p in parts]
    cand = os.sep.join(parts2)
    if os.path.exists(cand):
        return cand

    # try changing extension if file not found
    stem, _ = os.path.splitext(cand)
    for e in exts:
        cand2 = stem + e
        if os.path.exists(cand2):
            return cand2

    return cand  # fall back (will raise later)


def adjust_K_for_preprocess(
    K: np.ndarray,          # (3,3)
    W: int, H: int,         # original image size used by K
    new_width: int, new_height: int,
    mode: str = "crop",
    target_size: int = 518,
    # these are produced by your pipeline:
    start_y: int = 0,       # crop only (if used)
    pad_left: int = 0, pad_top: int = 0,  # pad only (if used)
) -> np.ndarray:
    K = K.astype(np.float64).copy()

    sx = new_width / float(W)
    sy = new_height / float(H)

    # 1) resize
    K[0, 0] *= sx   # fx
    K[1, 1] *= sy   # fy
    K[0, 2] *= sx   # cx
    K[1, 2] *= sy   # cy

    # 2) crop or pad
    if mode == "crop":
        # your code only crops vertically when new_height > target_size
        K[1, 2] -= float(start_y)
    elif mode == "pad":
        K[0, 2] += float(pad_left)
        K[1, 2] += float(pad_top)
    else:
        raise ValueError("mode must be 'crop' or 'pad'")

    return K

def load_and_preprocess_depths_and_valid_mask(
    image_path_list: List[str],
    depth_path_list: Optional[List[str]] = None,
    K: np.ndarray = None,
    all_poses: np.ndarray = None,
    mode: str = "crop",
    target_size: int = 518,
    make_divisible: int = 14,
    depth_dir: Optional[str] = None,
    rgb_dir_token: str = "rgb",
    depth_dir_token: str = "depth",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read depth images (same-name) and apply EXACT same geometric ops as load_and_preprocess_images:
      - compute resize dims based on RGB W,H and mode
      - resize depth with NEAREST (avoid creating fake positive depth)
      - crop/pad same way
      - if batch has different shapes, pad to max H,W (depth padded with 0)

    Returns:
      depths:      (N, 1, H, W) float32
      valid_masks: (N, 1, H, W) bool, where depth > 0

    Notes:
      - Depth is assumed already in "raw units" (e.g., mm). We keep values as-is.
        If you want meters, divide by 1000 outside.
      - For pad mode, we pad depth with 0 (invalid).
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    if depth_path_list is None:
        depth_path_list = [
            _infer_depth_path_from_rgb(
                p,
                depth_dir=depth_dir,
                rgb_dir_token=rgb_dir_token,
                depth_dir_token=depth_dir_token,
            )
            for p in image_path_list
        ]
    if len(depth_path_list) != len(image_path_list):
        raise ValueError("depth_path_list must have the same length as image_path_list")

    depths = []
    shapes = set()
    Ks_new = [] 
    # Load camera parameters and depth maps
    gt_extrinsics = []
    gt_intrinsics = []
    gt_depths = []
    gt_world_points = []
    count = 0
    for rgb_path, depth_path in zip(image_path_list, depth_path_list):
        # --- Open RGB only for geometry (W,H) ---
        rgb = Image.open(rgb_path)
        dep = Image.open(depth_path)


        width, height = rgb.size
        start_y = 0
        pad_left = 0
        pad_top = 0
        # --- decide new_w, new_h EXACTLY like your image function ---
        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / make_divisible) * make_divisible
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / make_divisible) * make_divisible
        else:  # crop
            new_width = target_size
            new_height = round(height * (new_width / width) / make_divisible) * make_divisible

        # --- load depth ---
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth file not found: {depth_path}")


        # keep depth numeric; DO NOT convert to RGB
        # Many depth PNGs are I;16 / I;32 etc. We'll force to a single-channel integer image.
        if dep.mode not in ("I;16", "I", "F"):
            # fallback: convert to 32-bit integer
            dep = dep.convert("I")
        d_width, d_height = dep.size
        # --- resize depth with NEAREST ---
        dep = dep.resize((new_width, new_height), Image.Resampling.NEAREST)

        # --- to torch: (1, H, W) float32 ---
        to_tensor_u16 = TF.PILToTensor()   # 输出 uint8/uint16/int32，形状 (1,H,W)
        dep_t = to_tensor_u16(dep)        # (1,H,W)
        # 有些 depth PNG 可能会变成 (H,W) 或 (C,H,W)，这里统一成 (1,H,W)
        if dep_t.dim() == 2:
            dep_t = dep_t.unsqueeze(0)
        elif dep_t.dim() == 3 and dep_t.shape[0] != 1:
            dep_t = dep_t[:1]  # 极少数异常情况：只取第一通道

        dep_t = dep_t.float()             # (1,H,W) float32

        # --- crop height center if crop mode and new_h > target_size ---
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            dep_t = dep_t[:, start_y : start_y + target_size, :]

        # --- pad to square if pad mode ---
        if mode == "pad":
            h_padding = target_size - dep_t.shape[1]
            w_padding = target_size - dep_t.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                # pad depth with 0 (invalid)
                dep_t = torch.nn.functional.pad(
                    dep_t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
                )

        # ---- 用这里：更新K ----

        K_new = adjust_K_for_preprocess(
            K=K, W=d_width, H=d_height,
            new_width=new_width, new_height=new_height,
            mode=mode, target_size=target_size,
            start_y=start_y, pad_left=pad_left, pad_top=pad_top
        )
        Ks_new.append(K_new)
        
        
        pose = all_poses[count]
        # get ground-truth point clouds
        # Apply depth thresholding (same as scannet.py)
        depthmap = threshold_depth_map(dep_t, max_percentile=99, min_percentile=-1)

        # Convert pose to camera-to-world (same as scannet.py)
        # tmp_pose = closed_form_inverse_se3(pose[None])[0]
        tmp_pose = pose
        

        # Compute world coordinates from depth map
        # print("depthmap.shape",depthmap.shape)
        # print("tmp_pose.shape",tmp_pose.shape)
        # print("K_new.shape",K_new.shape)
        world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
            depthmap.squeeze().detach().cpu().numpy(), tmp_pose[:3,:], K_new, z_far=100.0
        )

        gt_world_points.append(world_coords_points)
        
        
        
        shapes.add((dep_t.shape[1], dep_t.shape[2]))
        depths.append(dep_t)
        count += 1
    # --- if different shapes in batch: pad to max H,W (same policy as your images) ---
    if len(shapes) > 1:
        print(f"Warning: Found depth shapes different across batch: {shapes}")
        max_height = max(s[0] for s in shapes)
        max_width = max(s[1] for s in shapes)

        padded = []
        for dep_t in depths:
            h_padding = max_height - dep_t.shape[1]
            w_padding = max_width - dep_t.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                dep_t = torch.nn.functional.pad(
                    dep_t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
                )
            padded.append(dep_t)
        depths = padded

    depths = torch.stack(depths, dim=0)  # (N,1,H,W)
    Ks_new = np.stack(Ks_new, 0)   # (N,3,3)
    # --- valid mask: depth > 0, must match final shape ---
    valid_masks = depths > 0
    
    

    

    # ensure single image shape is (1,1,H,W)
    if len(image_path_list) == 1 and depths.dim() == 3:
        depths = depths.unsqueeze(0)
        valid_masks = valid_masks.unsqueeze(0)

    return depths, valid_masks, Ks_new, np.stack(gt_world_points)

def load_gt_data(target_dir: str, depth_path_list: List[str]) -> Optional[Dict[str, Any]]:
    """
    Load GT camera poses and specified depth maps from the scene directory.

    Args:
        target_dir: scene directory (contains all_poses.npy, K_depth.txt, etc.)
        depth_path_list: list of depth png paths to load (you specify which frames)

    Returns:
        Dictionary containing GT poses, intrinsics, depth maps, world points, etc.
        or None if data missing / mismatch.
    """
   

    if depth_path_list is None or len(depth_path_list) == 0:
        raise ValueError("depth_path_list must be a non-empty list of depth image paths")

    all_poses_path = os.path.join(target_dir, "all_poses.npy")
    K_path = os.path.join(target_dir, "K_depth.txt")
    if not os.path.exists(all_poses_path) or not os.path.exists(K_path):
        raise ValueError(f"Missing GT files in {target_dir}: all_poses.npy or K_depth.txt")
        return None

    all_poses = np.load(all_poses_path).reshape(-1, 4, 4)  # (T,4,4), assumed world-to-camera
    intrinsics_d = np.loadtxt(K_path).reshape(3, 3).astype(np.float32)

    # --- try to infer frame ids from filenames like "000123.png" or "123.png"
    depth_paths = list(depth_path_list)
    frame_ids = []
    all_numeric = True
    for p in depth_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            frame_ids.append(int(stem))
        except Exception:
            all_numeric = False
            frame_ids.append(None)

    if all_numeric:
        # keep a deterministic order by id (optional but usually desired)
        order = np.argsort(frame_ids).tolist()
        depth_paths = [depth_paths[i] for i in order]
        frame_ids = [frame_ids[i] for i in order]

        max_id = max(frame_ids)
        if max_id >= len(all_poses):
            raise ValueError(f"max_id {max_id} >= len(all_poses) {len(all_poses)}")
            return None

        poses_w2c = all_poses[frame_ids]  # (N,4,4)
    else:
        # fallback: assume provided order corresponds to first N poses
        if len(depth_paths) > len(all_poses):
            raise ValueError(f"len(depth_paths) {len(depth_paths)} > len(all_poses) {len(all_poses)}")
            return None
        poses_w2c = all_poses[:len(depth_paths)]
        frame_ids = list(range(len(depth_paths)))

    # --- load
    gt_extrinsics_c2w = []
    gt_intrinsics = []
    gt_depths = []
    gt_world_points = []
    valid_masks = []

    for i, depth_path in enumerate(depth_paths):
        pose_w2c = poses_w2c[i]  # (4,4)
        K = intrinsics_d.copy()

        # depth: uint16(mm) -> float32(m)
        depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32) / 1000.0
        depthmap[~np.isfinite(depthmap)] = 0.0

        # resize to DEFAULT_IMAGE_SIZE (same as你的原逻辑)
        depthmap, K, (sx, sy) = resize_depth_and_intrinsics_crop(
            depthmap, K,
        )

        # valid mask per-frame
        valid_mask = depthmap > 0
        valid_masks.append(valid_mask)

        # thresholding (same as scannet.py)
        depthmap = threshold_depth_map(depthmap, max_percentile=99, min_percentile=-1)

        # w2c -> c2w
        camera_pose_c2w = closed_form_inverse_se3(pose_w2c[None])[0]  # (4,4)

        # backproject to world
        world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
            depthmap, camera_pose_c2w, K, z_far=100.0
        )

        gt_extrinsics_c2w.append(camera_pose_c2w[:3, :])  # (3,4)
        gt_intrinsics.append(K.astype(np.float32))
        gt_depths.append(depthmap.astype(np.float32))
        gt_world_points.append(world_coords_points.astype(np.float32))

    gt_data = {
        "frame_ids": np.array(frame_ids, dtype=np.int64),                # (N,)
        "gt_extrinsic": np.stack(gt_extrinsics_c2w, axis=0),             # (N,3,4)  c2w
        "gt_intrinsic": np.stack(gt_intrinsics, axis=0),                 # (N,3,3)
        "gt_depth": np.stack(gt_depths, axis=0),                         # (N,H,W)
        "gt_world_points": np.stack(gt_world_points, axis=0),            # (N,H,W,3)
        "depth_paths": depth_paths,
        "valid_mask": np.stack(valid_masks, axis=0),                     # (N,H,W)
    }

    print(f"Loaded GT data for {len(depth_paths)} frames")
    print(f"GT extrinsic shape: {gt_data['gt_extrinsic'].shape}")
    print(f"GT intrinsic shape: {gt_data['gt_intrinsic'].shape}")
    print(f"GT depth shape: {gt_data['gt_depth'].shape}")
    print(f"GT mask shape: {gt_data['valid_mask'].shape}")

    return gt_data


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


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

    # depth_map = depth_map.astype(float, copy=True)

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


def resize_depth_and_intrinsics_crop(
    depthmap: np.ndarray,
    intrinsics_3x3: np.ndarray,
    target_size: int = 518,
    align_to: int = 14,
):
    """
    与 load_and_preprocess_images(mode='crop') 保持一致的 depth 处理：
      - new_w = target_size
      - new_h = round(h0 * (new_w / w0) / align_to) * align_to
      - resize (NEAREST)
      - 若 new_h > target_size: 中心裁剪高度到 target_size
      - 同步更新 K（resize 缩放 + crop 主点平移）

    Args:
        depthmap: [H, W] float32, meters
        intrinsics_3x3: [3,3]
    Returns:
        depth_out: [H', W'] (W'==target_size; H'==target_size 若发生crop，否则为 new_h)
        K_new: [3,3]
        scales: (sx, sy)
    """
    assert depthmap.ndim == 2, depthmap.shape
    assert intrinsics_3x3.shape == (3, 3), intrinsics_3x3.shape

    h0, w0 = depthmap.shape
    new_w = int(target_size)
    new_h = int(round((h0 * (new_w / float(w0))) / align_to) * align_to)

    sx = new_w / float(w0)
    sy = new_h / float(h0)

    # 1) resize depth (NEAREST)
    depth_rs = cv2.resize(depthmap, (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    # 2) update intrinsics for resize
    K_new = intrinsics_3x3.astype(np.float32).copy()
    K_new[0, 0] *= sx  # fx
    K_new[1, 1] *= sy  # fy
    K_new[0, 2] *= sx  # cx
    K_new[1, 2] *= sy  # cy

    # 3) center crop height if needed (crop mode)
    if new_h > target_size:
        start_y = (new_h - target_size) // 2
        depth_rs = depth_rs[start_y:start_y + target_size, :]
        K_new[1, 2] -= float(start_y)  # crop shifts principal point

    return depth_rs, K_new, (sx, sy)


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    z_far: float = 100.0,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a depth map to world coordinates (HxWx3) given the camera extrinsic and intrinsic.
    Returns both the world coordinates and the intermediate camera coordinates,
    as well as a mask for valid depth.

    Args:
        depth_map (np.ndarray):
            Depth map of shape (H, W).
        extrinsic (np.ndarray):
            Extrinsic matrix of shape (3, 4), representing the camera pose in OpenCV convention (camera-to-world).
        intrinsic (np.ndarray):
            Intrinsic matrix of shape (3, 3).
        eps (float):
            Small epsilon for thresholding valid depth.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (world_coords_points, cam_coords_points, point_mask)

            - world_coords_points: (H, W, 3) array of 3D points in world frame.
            - cam_coords_points: (H, W, 3) array of 3D points in camera frame.
            - point_mask: (H, W) boolean array where True indicates valid (non-zero) depth.
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps
    if z_far > 0:
        point_mask = point_mask #& (depth_map < z_far)

    # Convert depth map to camera coordinates
    depth_map = depth_map / 1000.0
    
    
    # show_depthmap_matplotlib(depth_map, idx=0, title="depth frame 0")
    
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)
    # The extrinsic is camera-from-world, so invert it to transform camera->world
    cam_to_world_extrinsic = extrinsic #closed_form_inverse_se3(extrinsic[None])[0]
    # cam_to_world_extrinsic = extrinsic
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]
    # print('cam_coords_points',cam_coords_points)
    # print('t_cam_to_world',t_cam_to_world)
    # Apply the rotation and translation to the camera coordinates
    # world_coords_points = (
    #     np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world
    # ) # HxWx3, 3x3 -> HxWx3
    # world_coords_points = (
    #     np.dot(R_cam_to_world, cam_coords_points) + t_cam_to_world
    # ) 
    world_coords_points = np.einsum('ij,hwj->hwi', R_cam_to_world, cam_coords_points) + t_cam_to_world.reshape(1, 1, 3)
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world
    # print('world_coords_points',world_coords_points)
    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

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
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def save_horizontal_rgb(images: torch.Tensor, save_path: str):
    """
    images: torch.Tensor [N, 3, H, W], value in [0,1] or [0,255]
    save_path: output image path
    """
    assert images.ndim == 4 and images.shape[1] == 3, images.shape

    # 如果是 GPU tensor
    images = images.detach().cpu()

    # 如果是 [0,1]，转成 [0,255]
    if images.max() <= 1.0:
        images = images * 255.0

    images = images.byte()  # uint8

    # 横向拼接：在 width 维度 concat
    concat = torch.cat(list(images), dim=2)  # [3, H, N*W]

    # CHW -> HWC
    concat = concat.permute(1, 2, 0).numpy()

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(concat).save(save_path)