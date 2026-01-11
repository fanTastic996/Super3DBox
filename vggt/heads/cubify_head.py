import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch

from functools import partial
from torch import nn
from collections import OrderedDict
from cubifyanything.vit import ViT

import copy
import itertools
import math
import numpy as np
import torch
import torch.nn.functional as F
import warnings

from torch import nn

from cubifyanything.boxes import GeneralInstance3DBoxes, DepthInstance3DBoxes
from cubifyanything.instances import Instances3D
from cubifyanything.measurement import WhitenedDepthMeasurementInfo
from cubifyanything.pos import CameraRayEmbedding
from cubifyanything.transforms import euler_angles_to_matrix
from cubifyanything.preprocessor import Augmentor, Preprocessor
from cubifyanything.measurement import ImageMeasurementInfo, DepthMeasurementInfo
from cubifyanything.orientation import ImageOrientation, rotate_tensor, ROT_Z
from cubifyanything.sensor import SensorArrayInfo, SensorInfo, PosedSensorInfo
from scipy.spatial.transform import Rotation
from cubifyanything.batching import Sensors
from PIL import Image
from einops import rearrange


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def parse_transform_3x3_np(data):
    return torch.tensor(data.reshape(3, 3).astype(np.float32))

def parse_transform_4x4_np(data):
    return torch.tensor(data.reshape(4, 4).astype(np.float32))

def parse_size(data):
    return tuple(int(x) for x in data.decode("utf-8").strip("[]").split(", "))

def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    try:
        return src.to(dst)
    except:
        return src.to(dst.device)

def move_to_current_device(x, t):
    if isinstance(x, (list, tuple)):
        return [move_device_like(x_, t) for x_ in x]
    
    return move_device_like(x, t)

def move_input_to_current_device(batched_input: Sensors, t: torch.Tensor):
    # Assume only two levels of nesting for now.
    return { name: { name_: move_to_current_device(m, t) for name_, m in s.items() } for name, s in batched_input.items() }

class LayerNorm2D(nn.Module):
    def __init__(self, normalized_shape, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ln = norm_layer(normalized_shape) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        """
        x: N C H W
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x

class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)    

# Plain-DETR.
class GlobalCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rpe_hidden_dim=512,
        rpe_type='linear',
        feature_stride=16,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rpe_type = rpe_type
        self.feature_stride = feature_stride

        # FFN.
        self.cpb_mlp1 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)
        self.cpb_mlp2 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        cpb_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, out_dim, bias=False))
        return cpb_mlp

    def forward(
        self,
        query,
        reference_2d,
        k_input_flatten,
        v_input_flatten,
        input_spatial_shapes,
        input_padding_mask=None,
        box_attn_prior_mask=None
    ):
        assert input_spatial_shapes.size(0) == 1, 'This is designed for single-scale decoder.'
        h, w = input_spatial_shapes[0]
        stride = self.feature_stride

        ref_2d_xyxy = torch.cat([
            reference_2d[:, :, :, :2] - reference_2d[:, :, :, 2:] / 2,
            reference_2d[:, :, :, :2] + reference_2d[:, :, :, 2:] / 2,
        ], dim=-1)  # B, nQ, 1, 4

        pos_x = torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=w.device)[None, None, :, None] * stride  # 1, 1, w, 1
        pos_y = torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=h.device)[None, None, :, None] * stride  # 1, 1, h, 1

        if self.rpe_type == 'abs_log8':
            delta_x = ref_2d_xyxy[..., 0::2] - pos_x  # B, nQ, w, 2
            delta_y = ref_2d_xyxy[..., 1::2] - pos_y  # B, nQ, h, 2
            delta_x = torch.sign(delta_x) * torch.log2(torch.abs(delta_x) + 1.0) / np.log2(8)
            delta_y = torch.sign(delta_y) * torch.log2(torch.abs(delta_y) + 1.0) / np.log2(8)
        elif self.rpe_type == 'linear':
            delta_x = ref_2d_xyxy[..., 0::2] - pos_x  # B, nQ, w, 2
            delta_y = ref_2d_xyxy[..., 1::2] - pos_y  # B, nQ, h, 2
        else:
            raise NotImplementedError

        rpe_x, rpe_y = self.cpb_mlp1(delta_x), self.cpb_mlp2(delta_y)  # B, nQ, w/h, nheads
        rpe = (rpe_x[:, :, None] + rpe_y[:, :, :, None]).flatten(2, 3) # B, nQ, h, w, nheads ->  B, nQ, h*w, nheads
        rpe = rpe.permute(0, 3, 1, 2)

        B_, N, C = k_input_flatten.shape
        k = self.k(k_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B_, N, C = query.shape
        q = self.q(query).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale

        attn = q @ k.transpose(-2, -1)

        # Only apply the RPE if the prior mask informs us to. Certain queries (like metric tokens) do not need priors.
        if box_attn_prior_mask is not None:
            attn[:, :, box_attn_prior_mask] += rpe
        else:
            attn += rpe

        if input_padding_mask is not None:
            attn += input_padding_mask[:, None, None] * -100

        fmin, fmax = torch.finfo(attn.dtype).min, torch.finfo(attn.dtype).max
        torch.clip_(attn, min=fmin, max=fmax)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# Plain-DETR.
class PreNormGlobalDecoderLayer(nn.Module):
    def __init__(
        self,
        xattn,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation=nn.ReLU,
        n_heads=8
    ):
        super().__init__()

        self.xattn = xattn

        if self.xattn is not None:
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        query_pos,
        reference_2d,
        src,
        src_pos_embed,
        src_spatial_shapes,
        src_padding_mask=None,
        self_attn_mask=None,
        box_attn_prior_mask=None
    ):
        # self attention
        tgt2 = self.norm2(tgt)

        q = k = self.with_pos_embed(tgt2, query_pos)

        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt2.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)

        # global cross attention
        if self.xattn is not None:
            tgt2 = self.norm1(tgt)
            tgt2 = self.xattn(
                self.with_pos_embed(tgt2, query_pos),
                reference_2d,
                self.with_pos_embed(src, src_pos_embed),
                src,
                src_spatial_shapes,
                src_padding_mask,
                box_attn_prior_mask=box_attn_prior_mask
            )

            tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)

        return tgt

def _rodrigues_from_axis_angle(axis, angle):
    # axis: (3,)  单位向量
    # angle: 标量 (rad)
    K = torch.tensor([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]], dtype=axis.dtype, device=axis.device)
    I = torch.eye(3, dtype=axis.dtype, device=axis.device)
    s, c = torch.sin(angle), torch.cos(angle)
    return I + s * K + (1 - c) * (K @ K)

def _align_up_remove_roll_pitch(R, eps=1e-8):
    """
    给定 3x3 旋转矩阵 R（把相机坐标系带到世界坐标系），
    返回一个 3x3 的校正矩阵 R_align，使得：
      - R_align 只由 x/z 自由度构成（轴 ∈ xz 平面，绕 y 的分量为 0）
      - 应用后，上方向对齐世界 +Y：  R_align @ (R @ e_y) = [0,1,0]
    """
    dtype, device = R.dtype, R.device
    I = torch.eye(3, dtype=dtype, device=device)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

    # 相机上方向在世界坐标下的方向：相机的 e_y 为 [0,1,0]，在世界系中是 R @ e_y = R[:,1]
    u = R[:, 1]  # (3,)
    u = u / (u.norm() + eps)

    # 旋转轴 = u × world_up，天然在 xz 平面（其 y 分量为 0），因此不含绕 y 的分量
    axis = torch.cross(u, world_up)
    s = axis.norm()
    c = torch.clamp((u * world_up).sum(), -1.0, 1.0)  # = cos(theta)

    if s < 1e-7:
        # u 与 world_up 平行或反平行
        if c > 0.0:
            # 已对齐：无需旋转
            R_align = I
        else:
            # 反向：需要 180°，任选 xz 平面内且与 u 垂直的轴
            # 取更稳定的：选 u 在 x 或 z 分量更小的正交轴
            if abs(u[0]) < abs(u[2]):
                axis180 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
            else:
                axis180 = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
            R_align = -I + 2.0 * (axis180[:, None] @ axis180[None, :])  # 180°: R = 2uu^T - I
    else:
        axis = axis / s
        angle = torch.atan2(s, c)  # 稳定地得到角度
        R_align = _rodrigues_from_axis_angle(axis, angle)

    return R_align

    
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

    # # this gets applied _after_ predictions to put it in camera space.
    # T = Rotation.from_euler("xz", Rotation.from_matrix(fake_basis[-1].cpu().numpy()).as_euler("yxz")[1:]).as_matrix()
    # # warning: /home/lanyuqing/myproject/vggt/vggt/heads/cubify_head.py:1692: UserWarning: Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.


    # return torch.tensor(T).to(pose)
    
    # TODO: change from_euler not sure if this is right
    R = fake_basis[-1].to(pose)             # (3,3)
    R_align = _align_up_remove_roll_pitch(R)  # (3,3)

    T = torch.eye(4, dtype=pose.dtype, device=pose.device)
    T[:3, :3] = R_align

    return T

        
@torch.jit.script_if_tracing
def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Tracing friendly way to cast tensor to another tensor's device. Device will be treated
    as constant during tracing, scripting the casting process as whole can workaround this issue.
    """
    return src.to(dst)

def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)

    return valid_ratio


class CubifyHead(nn.Module):
    """3D目标检测/分割模型的核心模块，整合了特征提取、位置编码、提示机制和解码器"""
    
    def __init__(
        self,
        backbone,       # 特征提取主干网络（如ViT）
        prompting,      # 提示生成机制组件
        decoder,        # 解码器（包含注意力机制和预测头）
        pixel_mean,     # 图像归一化均值
        pixel_std,      # 图像归一化标准差
        pos_embedding,  # 位置编码模块
        # fusion_module,  # 特征融合模块
        # vggt_merger,  # VGGT特征合并器
        # frame_merger, # 帧特征合并器
        sensor_name="wide", # 使用的传感器类型
        topk_per_image=100  # 每张图像保留的最大预测数
    ):
        super().__init__()
        # 基础组件初始化
        self.backbone = backbone
        self.prompting = prompting
        self.decoder = decoder
        self.pos_embedding = pos_embedding
        #added
        # self.fusion_module = fusion_module  # 特征融合模块
        # self.vggt_merger = vggt_merger  # VGGT特征合并器
        # self.frame_merger = frame_merger  # 帧特征合并器
        # 注册图像归一化参数（自动跟随模型设备移动）
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert self.pixel_mean.shape == self.pixel_std.shape, "归一化参数形状不匹配"

        # 多尺度特征投影
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone.num_channels[0], decoder.embed_dim, kernel_size=1),  # 通道维度转换
                nn.GroupNorm(32, decoder.embed_dim),  # 分组归一化
            )
        ])
        self.level_embed = nn.Parameter(torch.Tensor(len(backbone.num_channels), decoder.embed_dim))  # 多尺度级别嵌入
        self.sensor_name = sensor_name
        self.topk_per_image = topk_per_image
        
        is_depth_model = False
        self.augmentor = Augmentor(("wide/image", "wide/depth") if is_depth_model else ("wide/image",))
        self.preprocessor = Preprocessor()
        
        self.init_sensor()
    
    #
    def init_sensor(self):
        wide = PosedSensorInfo()            
        image_info = ImageMeasurementInfo(
            # size=(self.img_width, self.img_height),
            size=(518, 518), # H W
            # size=(1024, 768), # work
            K=torch.tensor([[573.6569,   0.0000, 259.0],
         [  0.0000, 575.0908, 259.0],
         [  0.0000,   0.0000,   1.0000]])[None])
        wide.image = image_info
        new_size = (int(wide.image.size[0] ), int(wide.image.size[1] ))
        wide.image = wide.image.resize(new_size)
        wide.RT = torch.eye(4)[None]
        current_orientation = wide.orientation
        target_orientation = ImageOrientation.UPRIGHT
        T_gravity = get_camera_to_gravity_transform(wide.RT[-1], current_orientation, target=target_orientation)
        wide.T_gravity = T_gravity[None]
        
        gt = PosedSensorInfo()        
        gt.RT = parse_transform_4x4_np(np.eye(4))[None]
        
        self.sensor_info = SensorArrayInfo()
        self.sensor_info.wide = wide
        self.sensor_info.gt = gt
    
    @property
    def device(self):
        """返回模型所在设备（通过pixel_mean自动获取）"""
        return self.pixel_mean.device

    def _flatten(self, srcs, pos_embeds, masks):
        """将多尺度特征展平为适合Transformer的序列格式
        Args:
            srcs: 多尺度特征图列表 [ (B, C, H, W) ]
            pos_embeds: 位置编码列表
            masks: 掩码列表
        Returns:
            src_flatten: 展平的特征序列 (B, L, C)
            ... # 其他返回参数见下方
        """
        src_flatten, mask_flatten, lvl_pos_embed_flatten = [], [], []
        spatial_shapes = []  # 存储各尺度特征图空间尺寸
        
        # 遍历每个尺度的特征
        for lvl, (src, pos_embed, mask) in enumerate(zip(srcs, pos_embeds, masks)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            # 展平特征和位置编码 (B, C, H, W) -> (B, H*W, C)
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)  # 展平掩码
            # 添加层级嵌入信息
            # lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed = pos_embed + self.level_embed[0].view(1, 1, -1)
            
            # 收集结果
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        
        # 拼接所有尺度的特征/位置编码/掩码
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        
        # 生成多尺度特征的索引信息
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)  # 有效区域比例
        
        return src_flatten, lvl_pos_embed_flatten, mask_flatten, spatial_shapes, level_start_index, valid_ratios

    

    
    # cubify没有camera pose，因此不应该按照cubify那样的，因为我们没有Pose
    def inference(
        self,
        vggt_images,  # 批处理的传感器数据字典 # vggt images: ([4(B), 3(N), 3, 476, 518])
        aggregated_tokens_list, #N个[266, 2048]的list
        patch_start_idx,
        intrinsic=None,
        extrinsic=None,
        gravity=None,
    ):        
        self.init_sensor()
        all_results=[]
        
        N_batch = vggt_images.shape[0]
        N_img = vggt_images.shape[1]
        img_H = vggt_images.shape[-2]
        img_W = vggt_images.shape[-1]
        #sensor_info: .wide.image .wide.image.K .wide.T_gravity .wide.RT
        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        all_packaged = []

        for j in range(N_batch): # 一个个batch轮流来，因为cubify不支持
        #     # 1.gravity 特征提取（主干网络）
            sample = {"wide": {'image': vggt_images[j]},
                      "sensor_info": self.sensor_info}
            
            sample["wide"]['image'] = (sample["wide"]['image'] * 255.).to(torch.uint8)

            if intrinsic is not None and extrinsic is not None:
                sample["sensor_info"].wide.image = sample["sensor_info"].wide.image.resize((img_H, img_W))  #TODO: changed img_H, img_W
                sample["sensor_info"].wide.image.K[0,:3,:3] = intrinsic[j,0].detach().cpu()[:3,:3]
            
            # GT T_gravity
            # sample["sensor_info"].wide.T_gravity = gravity[j]
            
            packaged = self.augmentor.package(sample)
            device = self.pixel_mean
            packaged = move_input_to_current_device(packaged, device)
            all_packaged.append(packaged)
            
        batched_sensors = self.preprocessor.preprocess(all_packaged)
        
        sensor = batched_sensors[self.sensor_name]   
        if len(sensor['image'].data.tensor.shape) > 4:
            sensor['image'].data.tensor = sensor['image'].data.tensor.squeeze(0) 
        if len(sensor['image'].data.tensor.shape) > 4:
            sensor['image'].data.tensor = sensor['image'].data.tensor.view(N_batch*N_img, 3, sensor['image'].data.tensor.shape[-2], sensor['image'].data.tensor.shape[-1])
        
        #TODO:
        '''
        sensor['image'].data.image_sizes改成有多少张图就多少个image_sizes
        sensor['image'].sensor是一个list[],每个元素代表一张图像，需要将VGGT的pose赋值到RT
        sensor['image'].sensor[index].image是一个ImageMeasurementInfo,需要将该变量的.K更改为每张图的实际内参[1,3,3]
        '''
        sensor['image'].data.image_sizes = sensor['image'].data.image_sizes * N_img #不确定这里要不要乘batch
        # base_sensor = sensor["image"].sensor[0]
        # for j in range(N_batch):
        #     sensor["image"].sensor[j].RT = sensor["image"].sensor[j].RT.repeat(N_img,1,1)
        #     sensor["image"].sensor[j].image.K = sensor["image"].sensor[j].image.K.repeat(N_img,1,1)
        #     sensor["image"].sensor[j].RT[:,:3,:4] = extrinsic[j, ...]
        #     sensor["image"].sensor[j].image.K = intrinsic[j]
            
        features = self.backbone(sensor)

        

        #sensor: image: 
        # data: tensor: [4, 3, H, W]  image_sizes: [(518, 518), (518, 518)]
        # info: [<cubifyanything.measurement.ImageMeasurementInfo object at 0x7f300c1adae0>, <cubifyanything.measurement.ImageMeasurementInfo object at 0x7f3036154040>]: K/size
        # 2. 准备多尺度特征
        srcs, masks, pos_embeds = [], [], []
        for l, (feat, info_) in enumerate(zip(features, sensor["image"].info)):
            # 计算位置编码（融合相机参数）

            pos_embeds_ = self.pos_embedding(feat, sensor)
            pos_embeds.append(pos_embeds_)
            
            # 分离特征和掩码
            src, mask = feat.decompose()
            # 投影到解码器维度
            # srcs.append(self.input_proj[l](src))
            srcs.append(self.input_proj[0](src))
            masks.append(mask)
        
        # 3. 展平特征（适配Transformer结构）
        flattened = self._flatten(srcs, pos_embeds, masks)
        src_flatten, lvl_pos_embed, mask_flatten, spatial_shapes, level_start_index, valid_ratios = flattened  

        # 特征融合
        #extract VGGT增强的3D视觉特征
        # vggt_features = self.extract_image_vggt_embeds_3d_direct(aggregated_tokens_list[-2],patch_start_idx,img_H,img_W)  # [batch_size, num_frames, C, H, W]
        vggt_features = aggregated_tokens_list[-1][:, :, patch_start_idx:, :]

        # vggt_features = vggt_features.view(src_flatten.shape[0],-1,vggt_features.shape[-1])  
        
        # multiframe_fused_features = self.fusion_module(src_flatten, vggt_features, mask_flatten)
        
        fused_features = src_flatten # single frame
        # fused_features = multiframe_fused_features #.reshape(1, -1, 256)  #[1, N*single_img_token, 256]
        
        fused_features = rearrange(fused_features, '(b n) c d -> b (n c) d', b = N_batch, n = N_img)
        mask_flatten = mask_flatten.reshape(N_batch, -1)
        
        # valid_ratios = valid_ratios.reshape(N)
        

        # 4. 生成提示（Prompt）
        '''
        Prompter的核心作用是根据图像特征生成​​物体查询（Object Queries）​​，这些查询作为Decoder的输入，承载了模型对场景中“可能存在哪些物体”的初始假设。其本质是一种​​物体候选生成器​​，类似于2D检测中的Region Proposal Network（RPN）
        '''
        # Mask_flatten 标识特征图中哪些位置是有效数据（非填充区域）与src_flatten相对应
        # valid_ratios ​有效图像区域相对于填充后图像尺寸的比例
    
        
        embedding_dim = fused_features.shape[-1]
        # generate box 2d proposals and queries
        prompts = self.prompting.get_image_prompts(
            fused_features, mask_flatten, spatial_shapes, sensor, N_batch, N_img, embedding_dim
        ) 
        prompters = self.prompting.prompters
        # 5. 解码器处理（核心预测流程）
        '''
        Decoder接收Prompter生成的Object Queries，通过与图像特征的交互（注意力机制）逐步优化这些查询，最终输出精确的3D检测结果
        '''
        lvl_pos_embed = lvl_pos_embed.repeat(1, N_img, 1)
        
        _, intermediate_preds = self.decoder(
            vggt_features,
            fused_features, lvl_pos_embed, mask_flatten, spatial_shapes, 
            level_start_index, valid_ratios, prompts, sensor, N_batch, N_img
        )
        
        # 6. 处理最后层输出
        # layer_number = len(intermediate_preds)
        
        prompt_outputs = intermediate_preds[-1] #[-1] old
        prompt_start_idx = 0
        results = None
        
        # 7. 遍历所有提示器生成最终预测
        for prompter_index, (prompt, prompter) in enumerate(zip(prompts, prompters)):
            # 提取当前提示器对应的输出
            prompt_outputs_ = [
                pred[prompt_start_idx:prompt_start_idx+prompt.number_prompts] 
                for pred in prompt_outputs
            ]
            
            # 仅处理需要输出的提示
            if prompt.has_output:
                # 调用提示器的推理方法（如框预测、分割等）
                results = prompter.inference( # only encoderproposals prompter since metric queries have already been done in previous steps.
                    prompt_outputs_, sensor, 
                    self.topk_per_image,
                    intrinsic, extrinsic, gravity
                )
                prompt_start_idx += prompt.number_prompts
                
                all_results = results # List: len=N_batch
                break  # 通常只有一个提示器产生输出
                
   
        return all_results


    def train_step(
        self,
        vggt_images,  # 批处理的传感器数据字典 # vggt images: ([4(B), 3(N), 3, 476, 518])
        aggregated_tokens_list, #N个[266, 2048]的list
        patch_start_idx,
        intrinsic=None,
        extrinsic=None,
        gravity=None,
    ):        
        self.init_sensor()
        all_results=[]
        
        N_batch = vggt_images.shape[0]
        N_img = vggt_images.shape[1]
        img_H = vggt_images.shape[-2]
        img_W = vggt_images.shape[-1]
        #sensor_info: .wide.image .wide.image.K .wide.T_gravity .wide.RT
        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        all_packaged = []

        for j in range(N_batch): # 一个个batch轮流来，因为cubify不支持
        #     # 1.gravity 特征提取（主干网络）
            sample = {"wide": {'image': vggt_images[j]},
                      "sensor_info": self.sensor_info}
            
            sample["wide"]['image'] = (sample["wide"]['image'] * 255.).to(torch.uint8)

            if intrinsic is not None and extrinsic is not None:
                sample["sensor_info"].wide.image = sample["sensor_info"].wide.image.resize((img_H, img_W))  #TODO: changed img_H, img_W
                sample["sensor_info"].wide.image.K[0,:3,:3] = intrinsic[j,0].detach().cpu()[:3,:3]
            
            # GT T_gravity
            # sample["sensor_info"].wide.T_gravity = gravity[j]
            
            packaged = self.augmentor.package(sample)
            device = self.pixel_mean
            packaged = move_input_to_current_device(packaged, device)
            all_packaged.append(packaged)
            
        batched_sensors = self.preprocessor.preprocess(all_packaged)
        
        sensor = batched_sensors[self.sensor_name]   
        if len(sensor['image'].data.tensor.shape) > 4:
            sensor['image'].data.tensor = sensor['image'].data.tensor.squeeze(0) 
        if len(sensor['image'].data.tensor.shape) > 4:
            sensor['image'].data.tensor = sensor['image'].data.tensor.view(N_batch*N_img, 3, sensor['image'].data.tensor.shape[-2], sensor['image'].data.tensor.shape[-1])
        
        #TODO:
        '''
        sensor['image'].data.image_sizes改成有多少张图就多少个image_sizes
        sensor['image'].sensor是一个list[],每个元素代表一张图像，需要将VGGT的pose赋值到RT
        sensor['image'].sensor[index].image是一个ImageMeasurementInfo,需要将该变量的.K更改为每张图的实际内参[1,3,3]
        '''
        sensor['image'].data.image_sizes = sensor['image'].data.image_sizes * N_img #不确定这里要不要乘batch
        # base_sensor = sensor["image"].sensor[0]
        # for j in range(N_batch):
        #     sensor["image"].sensor[j].RT = sensor["image"].sensor[j].RT.repeat(N_img,1,1)
        #     sensor["image"].sensor[j].image.K = sensor["image"].sensor[j].image.K.repeat(N_img,1,1)
        #     sensor["image"].sensor[j].RT[:,:3,:4] = extrinsic[j, ...]
        #     sensor["image"].sensor[j].image.K = intrinsic[j]
            
        features = self.backbone(sensor)

        

        #sensor: image: 
        # data: tensor: [4, 3, H, W]  image_sizes: [(518, 518), (518, 518)]
        # info: [<cubifyanything.measurement.ImageMeasurementInfo object at 0x7f300c1adae0>, <cubifyanything.measurement.ImageMeasurementInfo object at 0x7f3036154040>]: K/size
        # 2. 准备多尺度特征
        srcs, masks, pos_embeds = [], [], []
        for l, (feat, info_) in enumerate(zip(features, sensor["image"].info)):
            # 计算位置编码（融合相机参数）

            pos_embeds_ = self.pos_embedding(feat, sensor)
            pos_embeds.append(pos_embeds_)
            
            # 分离特征和掩码
            src, mask = feat.decompose()
            # 投影到解码器维度
            # srcs.append(self.input_proj[l](src))
            srcs.append(self.input_proj[0](src))
            masks.append(mask)
        
        # 3. 展平特征（适配Transformer结构）
        flattened = self._flatten(srcs, pos_embeds, masks)
        src_flatten, lvl_pos_embed, mask_flatten, spatial_shapes, level_start_index, valid_ratios = flattened  

        # 特征融合
        # extract VGGT增强的3D视觉特征
        # vggt_features = self.extract_image_vggt_embeds_3d_direct(aggregated_tokens_list[-2],patch_start_idx,img_H,img_W)  # [batch_size, num_frames, C, H, W]
        vggt_features = aggregated_tokens_list[-1][:, :, patch_start_idx:, :]


        # vggt_features = vggt_features.view(src_flatten.shape[0],-1,vggt_features.shape[-1])  
        
        # multiframe_fused_features = self.fusion_module(src_flatten, vggt_features, mask_flatten)
        
        fused_features = src_flatten # single frame
        # fused_features = multiframe_fused_features #.reshape(1, -1, 256)  #[1, N*single_img_token, 256]
        
        fused_features = rearrange(fused_features, '(b n) c d -> b (n c) d', b = N_batch, n = N_img)
        mask_flatten = mask_flatten.reshape(N_batch, -1)
        
        # valid_ratios = valid_ratios.reshape(N)
        

        # 4. 生成提示（Prompt）
        '''
        Prompter的核心作用是根据图像特征生成​​物体查询（Object Queries）​​，这些查询作为Decoder的输入，承载了模型对场景中“可能存在哪些物体”的初始假设。其本质是一种​​物体候选生成器​​，类似于2D检测中的Region Proposal Network（RPN）
        '''
        # Mask_flatten 标识特征图中哪些位置是有效数据（非填充区域）与src_flatten相对应
        # valid_ratios ​有效图像区域相对于填充后图像尺寸的比例
    
        
        embedding_dim = fused_features.shape[-1]
        # generate box 2d proposals and queries
        prompts = self.prompting.get_image_prompts(
            fused_features, mask_flatten, spatial_shapes, sensor, N_batch, N_img, embedding_dim
        ) 
        prompters = self.prompting.prompters
        # 5. 解码器处理（核心预测流程）
        '''
        Decoder接收Prompter生成的Object Queries，通过与图像特征的交互（注意力机制）逐步优化这些查询，最终输出精确的3D检测结果
        '''
        lvl_pos_embed = lvl_pos_embed.repeat(1, N_img, 1)
        
        _, intermediate_preds = self.decoder(
            vggt_features,
            fused_features, lvl_pos_embed, mask_flatten, spatial_shapes, 
            level_start_index, valid_ratios, prompts, sensor, N_batch, N_img
        )
        
        # 6. 处理最后层输出
        # layer_number = len(intermediate_preds)
        
        prompt_outputs = intermediate_preds #[-1] old
        prompt_start_idx = 0
        results = None
        
        # 7. 遍历所有提示器生成最终预测
        for prompter_index, (prompt, prompter) in enumerate(zip(prompts, prompters)):
            # 提取当前提示器对应的输出
            prompt_outputs_ = [
                pred[prompt_start_idx:prompt_start_idx+prompt.number_prompts] 
                for pred in prompt_outputs
            ]
            
            # 仅处理需要输出的提示
            if prompt.has_output:
                # 调用提示器的推理方法（如框预测、分割等）
                for prompt_outputs__ in prompt_outputs_:
                    results = prompter.inference( # only encoderproposals prompter since metric queries have already been done in previous steps.
                                    prompt_outputs__, sensor, 
                                    self.topk_per_image,
                                    intrinsic, extrinsic, gravity
                                )
                    all_results.append(results)
                prompt_start_idx += prompt.number_prompts
                
                # all_results = results # List: len=N_batch
                break  # 通常只有一个提示器产生输出
                
   
        return all_results


    def forward(self, batched_inputs, aggregated_tokens_list, patch_start_idx, intrinsic=None, extrinsic=None, gravity=None, do_postprocess=True, training=True):
        """训练/推理的统一入口（实际调用推理函数）"""
        if training:
            return self.train_step(batched_inputs, aggregated_tokens_list, patch_start_idx, intrinsic=intrinsic, extrinsic=extrinsic, gravity=gravity)
        else:
            return self.inference(batched_inputs, aggregated_tokens_list, patch_start_idx, intrinsic=intrinsic, extrinsic=extrinsic, gravity=gravity)

    def extract_image_vggt_embeds_3d(
        self,
        aggregated_tokens_list,
        patch_start_idx,
        img_H,
        img_W,
        ith,
        # images_vggt # 输入图像序列 [batch_size, num_frames, C, H, W]
    ):
        """
        提取VGGT增强的3D视觉特征 (image_embeds_3d)
        返回形状: [N, 2048] (N = batch_size * num_frames * h_grid_after_merge * w_grid_after_merge)
        """

        image_embeds_3d = []  # 存储每张图像的3D特征
        
        # for i in range(batch_size):
        n_image = aggregated_tokens_list.shape[1]  # 当前batch的帧数
        height, width = img_H, img_W # 图像原始分辨率
        
        # 获取视觉配置参数
        patch_size = 14 #self.config.vision_config.patch_size
        merge_size = 2 #self.config.vision_config.spatial_merge_size
        
        # 计算分块网格尺寸
        h_grid, w_grid = height // patch_size, width // patch_size
        h_grid_after_merge = h_grid // merge_size
        w_grid_after_merge = w_grid // merge_size
        
        # 自动选择混合精度（根据GPU能力）
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
        # 2. VGGT特征提取
        # 使用聚合器提取特征
        # aggregated_tokens_list, patch_start_idx = self.vggt_model.aggregator(images_vggt[i][None])
        #TODO:好像有问题，怎么就只取[-2]呢 #patch_start_idx: 5
        # print("len()",len(aggregated_tokens_list),aggregated_tokens_list[0].shape) #len() 4 torch.Size([3, 1263, 2048])
        # print("aggregated_tokens_list[-2]",aggregated_tokens_list[-2].shape)
        features = aggregated_tokens_list[ith, :, patch_start_idx:]
        
        # 4. 空间特征重组
        # print("features",features.shape)
        # print('debug',n_image,h_grid, w_grid)
        features = features.view(n_image, h_grid, w_grid, -1)
        features = features[:, :h_grid_after_merge * merge_size, :w_grid_after_merge * merge_size, :].contiguous()
        features = features.view(n_image, h_grid_after_merge, merge_size, w_grid_after_merge, merge_size, -1)
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous()
        # 5. 通过PatchMerger降采样（关键步骤）
        features = self.vggt_merger(features)#.to(self.visual.dtype))
        image_embeds_3d.append(features)
        
        # 6. 批量特征拼接
        return torch.cat(image_embeds_3d, dim=0)#.to(self.visual.dtype)

    def extract_image_vggt_embeds_3d_direct(
        self,
        aggregated_tokens_list,
        patch_start_idx,
        img_H,
        img_W,
        # images_vggt # 输入图像序列 [batch_size, num_frames, C, H, W]
    ):
        """
        提取VGGT增强的3D视觉特征 (image_embeds_3d)
        返回形状: [N, 2048] (N = batch_size * num_frames * h_grid_after_merge * w_grid_after_merge)
        """

        image_embeds_3d = []  # 存储每张图像的3D特征
        
        # for i in range(batch_size):
        n_image = aggregated_tokens_list.shape[1]  # 当前batch的帧数
        n_batch = aggregated_tokens_list.shape[0] 
        height, width = img_H, img_W # 图像原始分辨率
        
        # 获取视觉配置参数
        patch_size = 14 #self.config.vision_config.patch_size
        merge_size = 2 #self.config.vision_config.spatial_merge_size
        
        # 计算分块网格尺寸
        h_grid, w_grid = height // patch_size, width // patch_size
        h_grid_after_merge = h_grid // merge_size
        w_grid_after_merge = w_grid // merge_size
        
        # 自动选择混合精度（根据GPU能力）
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
        features = aggregated_tokens_list[:, :, patch_start_idx:]
        
        # 4. 空间特征重组
        # features的维度为[N_batch, N_image, tokens, feature_dim]
        features = features.view(n_image*n_batch, h_grid, w_grid, -1)
        features = features[:, :h_grid_after_merge * merge_size, :w_grid_after_merge * merge_size, :].contiguous()
        features = features.view(n_image*n_batch, h_grid_after_merge, merge_size, w_grid_after_merge, merge_size, -1)
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous()
        # 5. 通过PatchMerger降采样（关键步骤）
        features = self.vggt_merger(features)#.to(self.visual.dtype))
        # image_embeds_3d.append(features)
        
        # 6. 批量特征拼接,最后的
        # return torch.cat(image_embeds_3d, dim=0)#.to(self.visual.dtype)
        return features
    
if __name__ == "__main__":
    
    cubify_model = '/home/lyq/myprojects/VGGT-cubify/models/cutr_rgbd.pth'
    checkpoint = torch.load(cubify_model, map_location='cuda' or "cpu")["model"]
    print(f"Loaded model from {cubify_model}")
    
    is_depth_model = 'rgbd' in cubify_model
    backbone_embedding_dimension=768
    
    vggt = make_vggt()
    model = make_vggt_cubify_transformer(vggt, dimension=backbone_embedding_dimension, depth_model=is_depth_model).eval()
    
    # 过滤掉不需要的参数（通过 key 匹配）
    model_dict = model.state_dict()
    filtered_dict = OrderedDict()

    for k in checkpoint.keys():
        # 跳过包含 vggt_model 或 fusion_model 的键
        if not (k.startswith("vggt_model") or k.startswith("fusion_model") or k.startswith("vggt_merger")):
            if k in model_dict:  # 确保当前模型存在该参数
                filtered_dict[k] = checkpoint[k]
            else:
                print(f"Warning: Key {k} not found in current model")
    
    # 加载筛选后的参数
    model.load_state_dict(filtered_dict, strict=False)  # strict=False 允许部分加载
    
    '''
    *******************************************************************
    '''

    #VGGT pre-process
    device = "cuda"
    image_root='/home/lyq/myprojects/VG-LLM/data/demo_data/3d_video_object_detection/media/'
    images_path = os.path.join(image_root, '0*.jpg')
    images_path_list = glob.glob(images_path)
    print("images_path_list: ",images_path_list)
    

    patch_size = 14 #self.processor.image_processor.patch_size
    merge_size = 2 #self.processor.image_processor.merge_size
        
    # 图像预处理
    image = load_and_preprocess_images(images_path_list) #(N, 3, H, W) [3, 392, 532]
    image = image.to(device)  # 将图像移动到GPU
    images_vggt = [image[i] for i in range(image.shape[0])]
    print('images_vggt',images_vggt[0].shape,len(images_vggt)) 

