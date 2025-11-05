# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head            # 激活与通道拆分工具：按配置将输出拆成预测与置信度
from .utils import create_uv_grid, position_grid_to_embed  # 生成UV网格并映射到通道维度的简单位置编码

# =========================================
# DPTHead：用于密集预测任务的头部（Depth/Pointmap/Normal等）
# 典型输入：来自ViT的多层tokens列表 + 原图 + patch起始索引（跳过非patch token）
# 典型输出：feature_only=True时为融合后的特征；否则为(预测, 置信度)
# 多尺度融合采用RefineNet风格的自顶向下(Top-Down)融合
# =========================================
class DPTHead(nn.Module):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.
    """

    def __init__(
        self,
        dim_in: int,                               # ViT单个token的通道数(C)
        patch_size: int = 14,                      # ViT的patch大小（决定patch网格H/patch_size, W/patch_size）
        output_dim: int = 4,                       # 输出通道数：如pointmap可为3或4(含置信度)
        activation: str = "inv_log",               # 预测通道的激活方式
        conf_activation: str = "expp1",            # 置信度通道的激活方式
        features: int = 256,                       # 融合后中间特征通道数
        out_channels: List[int] = [256, 512, 1024, 1024],  # 四个尺度的投影通道数
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],  # 选取的ViT中间层索引（从aggregated tokens取）
        pos_embed: bool = True,                    # 是否叠加简易位置嵌入(基于UV网格)
        feature_only: bool = False,                # True则只返回特征，不输出头部预测
        down_ratio: int = 1,                       # 输出下采样倍数（=1时输出与输入同分辨率）
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)           # 对token通道做LayerNorm（在还原为特征图前）

        # Projection layers for each output channel from tokens.
        # 四个尺度对应的1×1卷积，将[C]投影到指定通道数（out_channels[i]）
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # Resize layers for upsampling feature maps.
        # 将四个尺度的特征对齐到相近空间尺度（分别使用反卷积×4、×2、恒等、stride=2的卷积）
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)  # 3×3 conv把四层通道统一到features

        # Attach additional modules to scratch.
        # 构造四级RefineNet式的特征融合模块（自顶向下逐层融合）
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)  # 顶层仅自处理，不与更高层相加

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            # 仅需要特征时：保留一层3×3卷积作为输出
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            # 需要预测头时：先降一半通道，再用两层卷积得到最终output_dim
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],  # 来自不同层的token列表，形如[List[B,S,N_token,C]]
        images: torch.Tensor,                        # 输入图像 [B, S, 3, H, W]，取其H,W与分块计算尺寸
        patch_start_idx: int,                        # patch token在序列中的起始位置（跳过非patch的先导token）
        frames_chunk_size: int = 8,                  # 按帧分块处理以节省显存，None或>=S则一次性处理
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W]
        """
        B, S, _, H, W = images.shape

        # 若不需要分块或块大小>=序列长度，直接全量处理
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # 否则按块处理
        assert frames_chunk_size > 0

        # 分块累积输出
        all_preds = []
        all_conf = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # 对该块进行前向
            if self.feature_only:
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else:
                chunk_preds, chunk_conf = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)

        # 沿序列维拼接回完整输出
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()  # 按块裁剪图像以匹配tokens切片

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size  # patch网格尺寸

        out = []
        dpt_idx = 0

        # 从多个中间层取patch tokens，恢复为空间特征并对齐尺度
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]  # 跳过非patch token

            # 若启用分块，则同步裁剪对应帧段
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1])          # [B*S, N_patch, C]
            x = self.norm(x)                                # 对token通道归一化
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))  # -> [B*S, C, Hp, Wp]

            x = self.projects[dpt_idx](x)                  # 1×1卷积投影到该层指定通道数
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)         # 叠加基于UV网格的位置编码（轻量）
            x = self.resize_layers[dpt_idx](x)             # 对齐到共同的金字塔尺度

            out.append(x)
            dpt_idx += 1

        # 多层特征融合（RefineNet样式的自顶向下）
        out = self.scratch_forward(out)
        '''
        res = self.resConfUnit1(xs[1])
        output = output + res      # 融合高低层特征
        output = self.resConfUnit2(output)
        output = interpolate(output, size=上一级尺寸)
        '''
        # 双线性插值到目标输出分辨率（可通过down_ratio控制是否下采样）
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)         # 输出端再次加入位置编码以增强空间感知

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])          # 仅返回融合特征 [B,S,C,H',W']

        # 预测头：3×3 -> ReLU -> 1×1，得到最终output_dim通道
        out = self.scratch.output_conv2(out)
        # 将输出分离为预测与置信度，并按配置做激活（例如深度用inv_log、置信度用expp1）
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        # 生成与x空间尺寸匹配的UV网格（考虑宽高比），再映射到通道维并缩放
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])      # 将2通道UV映射到与x通道数一致
        pos_embed = pos_embed * ratio                                  # 控制位置嵌入强度
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed                                           # 残差式叠加

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        # 四个尺度特征
        layer_1, layer_2, layer_3, layer_4 = features

        # 3×3卷积统一通道（不带BN）
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # 顶层先自处理（无残差分支），并上采样到下一层大小
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        # 逐级：与下一层做残差融合，再上采样到更低一层大小
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        # 若feature_only==True，这里就是最终输出；否则还会走output_conv2得到预测
        out = self.scratch.output_conv1(out)
        return out


################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    # 构造一个FeatureFusionBlock：包含(可选)残差单元 × 2 + 上采样 + 1×1卷积
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    # 将四个输入尺度的通道各自通过3×3卷积变换到统一的out_shape（或其倍数）
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        # 可选：更高层通道更宽（未启用）
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()  # 量化友好的加法（也可用于普通残差相加）

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)    # 激活 -> conv3×3 -> (可选norm)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)  # 再次激活 -> conv3×3 -> (可选norm)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)  # 残差相加


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)  # 跳连分支的残差单元1

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)      # 主分支的残差单元2

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]  # 主分支特征（来自上一层融合后的输出）

        if self.has_residual:
            res = self.resConfUnit1(xs[1])         # 跳连特征经过残差单元
            output = self.skip_add.add(output, res)  # 与主分支相加

        output = self.resConfUnit2(output)         # 再经过一个残差单元

        # 计算上采样参数：默认尺度×2；若提供size则对齐到给定空间大小
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        # 上采样 + 1×1卷积(压缩/整合通道)
        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    # 自定义插值函数：当张量规模过大时按batch维分块插值，避免F.interpolate内部的INT_MAX限制
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736  # 安全阈值

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]  # 估算插值操作的元素规模

    if input_elements > INT_MAX:
        # 超过阈值则按batch维切块分别插值，再拼接
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)