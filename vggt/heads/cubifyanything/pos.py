import math
import torch
import warnings

from torch import nn
from torch.nn import functional as F

from math import log2, pi

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list, sensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale
        else:
            y_embed = (y_embed - 0.5) * self.scale
            x_embed = (x_embed - 0.5) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos

def generate_rays(
    info, image_shape, noisy: bool = False
):
    camera_intrinsics = info.K[-1][None]
    batch_size, device, dtype = (
        camera_intrinsics.shape[0],
        camera_intrinsics.device,
        camera_intrinsics.dtype,
    )
    height, width = image_shape
    # Generate grid of pixel coordinates
    pixel_coords_x = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    pixel_coords_y = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if noisy:
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5
    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()], dim=2
    )  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5

    # Handle radial distortion.
    ray_is_valid = torch.ones((height, width), dtype=torch.bool, device=device)

    # Calculate ray directions
    intrinsics_inv = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics_inv[:, 0, 0] = 1.0 / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 1] = 1.0 / camera_intrinsics[:, 1, 1]
    intrinsics_inv[:, 0, 2] = -camera_intrinsics[:, 0, 2] / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 2] = -camera_intrinsics[:, 1, 2] / camera_intrinsics[:, 1, 1]
    homogeneous_coords = torch.cat(
        [pixel_coords, torch.ones_like(pixel_coords[:, :, :1])], dim=2
    )  # (H, W, 3)

    ray_directions = torch.matmul(
        intrinsics_inv, homogeneous_coords.permute(2, 0, 1).flatten(-2)).view(
            3, height, width).permute(1, 2, 0)  # (3, H*W)

    ray_directions = F.normalize(ray_directions, dim=-1)  # (B, 3, H*W)
    theta = torch.atan2(ray_directions[..., 0], ray_directions[..., -1])
    phi = torch.acos(ray_directions[..., 1])
    angles = torch.stack([theta, phi], dim=-1)

    # Ensure we set anything invalid to just 0?
    ray_directions[~ray_is_valid] = 0.0
    angles[~ray_is_valid] = 0.0

    return ray_directions, angles

def generate_fourier_features(
    x: torch.Tensor,
    dim: int = 256,
    max_freq: int = 64,
    use_cos: bool = False,
    use_log: bool = False,
    cat_orig: bool = False,
):
    x_orig = x
    device, dtype, input_dim = x.device, x.dtype, x.shape[-1]
    num_bands = dim // (2 * input_dim) if use_cos else dim // input_dim

    if use_log:
        scales = 2.0 ** torch.linspace(
            0.0, log2(max_freq), steps=num_bands, device=device, dtype=dtype
        )
    else:
        scales = torch.linspace(
            1.0, max_freq / 2, num_bands, device=device, dtype=dtype
        )

    x = x.unsqueeze(-1)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat(
        (
            [x.sin(), x.cos()]
            if use_cos
            else [
                x.sin(),
            ]
        ),
        dim=-1,
    )

    if cat_orig:
        raise NotImplementedError

    return x.flatten(3)    

# Adopted from UniDepth. I don't think this is necessary, but keeping until we re-train models.
class CameraRayEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.proj = nn.Linear(255, self.dim)

    def forward(self, tensor_list, sensor):
        x = tensor_list.tensors
        
        feat_size =  tensor_list.tensors.shape[-1]
        # Hard-coded stride.
        square_pad = feat_size * 16

        # Generate the rays for the original images.
        ray_dirs = []
        for info_ in sensor["image"].info:
            ray_dirs_, angles_ = generate_rays(info_, (info_.size[1], info_.size[0]))
            ray_dirs_ = F.pad(ray_dirs_, (0, 0, 0, square_pad - ray_dirs_.shape[1], 0, square_pad - ray_dirs_.shape[0]))
            ray_dirs.append(ray_dirs_)

        ray_dirs = torch.stack(ray_dirs)

        rays_embedding = F.interpolate(ray_dirs.permute(0, 3, 1, 2), (feat_size, feat_size), mode="nearest").permute(0, 2, 3, 1)
        rays_embedding = F.normalize(rays_embedding, dim=-1)
        rays_embedding = generate_fourier_features(
            rays_embedding,
            dim=self.dim,
            max_freq=feat_size // 2,
            use_log=True,
            cat_orig=False,
        )

        rays_embedding = self.proj(rays_embedding)
        return rays_embedding.permute(0, 3, 1, 2).contiguous()
