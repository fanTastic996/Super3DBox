# This is a self-contained version of Detectron2's ViT with additional modifications (only meant for inference).
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp
from typing import Union

from cubifyanything.batching import BatchedPosedSensor

__all__ = ["ViT"]

# NOTE: We replicate some functions here which need modifications for tracing.
def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    x = x[:, :H, :W, :].contiguous()

    return x

def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    new_abs_pos = F.interpolate(
        abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    )

    return new_abs_pos.permute(0, 2, 3, 1)

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768, bias=True
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        proj_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
        depth_modality=False,
        depth_input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # Not supported.
            raise NotImplementedError

        self.depth_modality = depth_modality

    def forward(self, x, depth=None):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        if self.depth_modality and (depth is not None):
            B, H_d, W_d, _ = depth.shape

            # qkv with shape (3, B, nHead, H * W, C)
            qkv_depth = self.qkv(depth).reshape(B, H_d * W_d, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

            # q, k, v with shape (B * nHead, H * W, C)
            q_d, k_d, v_d = qkv_depth.reshape(3, B * self.num_heads, H_d * W_d, -1).unbind(0)
            q, k, v = torch.cat((q, q_d), dim=1), torch.cat((k, k_d), dim=1), torch.cat((v, v_d), dim=1)

            # presumably, concatenate q, k. split (and then reconcatenate) attn.

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.depth_modality and (depth is not None):
            attn, attn_d = torch.split(attn, (H * W, H_d * W_d), dim=1)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        if self.depth_modality and (depth is not None):
            attn_d = attn_d.softmax(dim=-1)
            depth = (attn_d @ v).view(B, self.num_heads, H_d, W_d, -1).permute(0, 2, 3, 1, 4).reshape(B, H_d, W_d, -1)
            depth = self.proj(depth)

        x = self.proj(x)
        return x, depth

DEPTH_WINDOW_SIZES = [4, 8, 16]
class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True, 
        mlp_bias=True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
        depth_modality=False,
        depth_window_size=0,
        layer_scale=False
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()

        if depth_modality and (depth_window_size == 0):
            raise ValueError("unsupported")

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            depth_modality=depth_modality,
            depth_input_size=(depth_window_size, depth_window_size) if depth_modality else None,
        )

        self.ls1 = None
        self.ls2 = None
        
        if layer_scale:
            self.ls1 = LayerScale(dim, 1.)
            self.ls2 = LayerScale(dim, 1.)

        self.depth_modality = depth_modality

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, bias=mlp_bias)
        self.drop_path = nn.Identity()

        self.window_size = window_size
        self.depth_window_size = depth_window_size

    def forward(self, x, depth=None):
        shortcut = x

        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        if self.depth_modality and (depth is not None):
            shortcut_depth = depth
            depth = self.norm1(depth)

            H_depth, W_depth = depth.shape[1], depth.shape[2]

            # Aggressive checking for now.
            depth_window_size = self.depth_window_size or (self.window_size // (H / H_depth))
            if isinstance(depth_window_size, torch.Tensor):
                depth_window_size = depth_window_size.int()
                if not depth_window_size.item() in DEPTH_WINDOW_SIZES:
                    raise ValueError(f"Unexpected window size {depth_window_size}")
            else:
                depth_window_size = int(depth_window_size)
                if not depth_window_size in DEPTH_WINDOW_SIZES:
                    raise ValueError(f"Unexpected window size {depth_window_size}")

            # if depth_window_size is not given, dynamically compute it based on the RGB window size and relative scale.
            depth, pad_hw_depth = window_partition(depth, depth_window_size)

        x, depth = self.attn(x, depth=depth)

        if self.depth_modality and (depth is not None):
            if self.window_size > 0:
                depth = window_unpartition(depth, depth_window_size, pad_hw_depth, (H_depth, W_depth))

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        if self.ls1 is not None:
            x = self.ls1(x)
            if self.depth_modality and (depth is not None):
                depth = self.ls1(depth)

        x = shortcut + self.drop_path(x)
        shortcut = x
        x = self.mlp(self.norm2(x))

        if self.ls2 is not None:
            x = self.ls2(x)

        x = shortcut + self.drop_path(x)

        if self.depth_modality and (depth is not None):
            depth = shortcut_depth + self.drop_path(depth)
            shortcut_depth = depth
            depth = self.mlp(self.norm2(depth))
            if self.ls2 is not None:
                depth = self.ls2(depth)

            depth = shortcut_depth + self.drop_path(depth)

        return x, depth

class ViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=None,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        mlp_bias=True,
        patch_embed_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        gated_mlp=False,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
        depth_modality=False,
        depth_window_size=0,
        encoder_norm=False,
        layer_scale=False,
        image_name="image",
        depth_name="depth"
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.depth_modality = depth_modality

        self.image_name = image_name
        self.depth_name = depth_name

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=patch_embed_bias,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        self.pos_embed_depth = None
        if self.depth_modality:
            self.patch_embed_depth = PatchEmbed(
                kernel_size=(16, 16),
                stride=(16, 16),
                in_chans=1,
                embed_dim=embed_dim,
            )

            if use_abs_pos:
                # note, depth gets its own pos embed.
                # Initialize absolute positional embedding with pretrain image size.
                # at some point, this size may differ from RGB's size.
                num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
                num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
                self.pos_embed_depth = nn.Parameter(torch.zeros(1, num_positions, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias, 
                mlp_bias=mlp_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=img_size,
                depth_modality=depth_modality and (i in window_block_indexes), # (for now, only attend to depth if windowing)
                depth_window_size=depth_window_size if i in window_block_indexes else 0,
                layer_scale=layer_scale
            )

            self.blocks.append(block)

        self.encoder_norm = norm_layer(embed_dim) if encoder_norm else nn.Identity()

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]
        self.window_block_indexes = window_block_indexes

        self.drop_path = nn.Identity()

        self._square_pad = [256, 384, 512, 640, 768, 896, 1024, 1280]

    @property
    def num_channels(self):
        return list(self._out_feature_channels.values())

    @property
    def size_divisibility(self):
        return next(iter(self._out_feature_strides.values()))

    def forward(self, s: BatchedPosedSensor):
        x = s[self.image_name].data.tensor #'image'
        image_shape = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2]))

        has_depth = self.depth_name in s
        has_depth_dropped = self.depth_modality and not has_depth
            
        if self.depth_modality:
            depth = s[self.depth_name].data.tensor[:, None] #'depth'
            depth = self.patch_embed_depth(depth)
            if self.pos_embed_depth is not None:
                depth = depth + get_abs_pos(
                    self.pos_embed_depth, self.pretrain_use_cls_token, (depth.shape[1], depth.shape[2]))
        else:
            depth = None

        for i, blk in enumerate(self.blocks):
            if blk.depth_modality and has_depth:
                x, depth = blk(x, depth=depth)
            else:
                x, *_ = blk(x)

        x = self.encoder_norm(x)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs

