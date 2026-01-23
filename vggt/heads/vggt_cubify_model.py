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
from cubifyanything.measurement import ImageMeasurementInfo, DepthMeasurementInfo
from cubifyanything.orientation import ImageOrientation, rotate_tensor, ROT_Z
from scipy.spatial.transform import Rotation


import base64
from PIL import Image
import glob

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

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




def _infer_effective_hw_from_pad_mask(mask_1d: torch.Tensor, H_pad: int, W_pad: int):
    """
    根据 padding mask 推断 src_flatten 的“真实有效区域”大小 H_eff, W_eff。

    输入：
      mask_1d: [H_pad*W_pad]，bool，True 表示这个 token 是 padding，False 表示有效
      H_pad, W_pad: src_flatten 的“固定网格尺寸”，比如 40,40

    假设：
      有效 token 形成“左上角对齐”的矩形区域，padding 在右侧/下侧（很常见）

    输出：
      H_eff, W_eff：有效区域高宽
    """
    assert mask_1d.numel() == H_pad * W_pad
    pad2d = mask_1d.view(H_pad, W_pad)     # [H_pad, W_pad]
    valid2d = ~pad2d                       # True 表示有效 token

    # 哪些行/列存在至少一个有效 token
    row_has = valid2d.any(dim=1)           # [H_pad]
    col_has = valid2d.any(dim=0)           # [W_pad]

    # 全 padding 的极端情况
    if not row_has.any() or not col_has.any():
        return 0, 0

    # 最后一行/列出现有效 token 的位置 + 1 => 有效区域尺寸
    H_eff = int(row_has.nonzero(as_tuple=False)[-1].item() + 1)
    W_eff = int(col_has.nonzero(as_tuple=False)[-1].item() + 1)

    # 可选：检查有效区域内部是否真的全有效（不严格矩形时这里可能不全为 True）
    inner = valid2d[:H_eff, :W_eff]
    if not inner.all():
        # 这里不强制报错，只是提醒：你的有效 token 可能不是严格左上矩形
        # 如果你确定一定是矩形，也可以改成 raise RuntimeError(...)
        pass

    return H_eff, W_eff


def _top_left_rect_flat_indices(H_eff: int, W_eff: int, W_pad: int, device):
    """
    给定有效区域 H_eff x W_eff（左上角对齐），返回它在 40x40 flatten 后的索引列表。

    flatten 规则：row-major（按行展开）
      idx = y * W_pad + x

    输出：
      idx: [H_eff*W_eff]
    """
    ys = torch.arange(H_eff, device=device)[:, None]  # [H_eff,1]
    xs = torch.arange(W_eff, device=device)[None, :]  # [1,W_eff]
    idx = (ys * W_pad + xs).reshape(-1)               # [H_eff*W_eff]
    return idx


class DeformableSrcFuseVggt(nn.Module):
    def __init__(
        self,
        d_src: int = 256,
        d_vggt: int = 2048,
        d_model: int = 256,
        num_points: int = 8,
        hidden: int = 256,
        offset_scale: float = 2.0,
        align_corners: bool = True,
    ):
        super().__init__()
        self.d_src = d_src
        self.d_vggt = d_vggt
        self.d_model = d_model
        self.K = num_points
        self.offset_scale = float(offset_scale)
        self.align_corners = bool(align_corners)

        self.vggt_proj = nn.Linear(d_vggt, d_model)
        self.q_norm = nn.LayerNorm(d_src)

        self.offset_mlp = nn.Sequential(
            nn.Linear(d_src, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.K * 2),
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(d_src, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.K),
        )
        self.fuse_gate = nn.Sequential(
            nn.Linear(d_src + d_model, d_model),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(d_model, d_src) if d_model != d_src else nn.Identity()

        # cache: key -> tensor (on specific device)
        self._rect_cache = {}
        self._base_cache = {}

    @torch.no_grad()
    def _make_base_grid_norm(self, H: int, W: int, Hv: int, Wv: int, device):
        if H == 0 or W == 0:
            return torch.empty((0, 2), device=device)

        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]

        if H > 1:
            yv = yy.float() * (Hv - 1) / float(H - 1)
        else:
            yv = torch.zeros_like(yy, dtype=torch.float32)

        if W > 1:
            xv = xx.float() * (Wv - 1) / float(W - 1)
        else:
            xv = torch.zeros_like(xx, dtype=torch.float32)

        if self.align_corners:
            x_norm = 2.0 * (xv / max(Wv - 1, 1)) - 1.0
            y_norm = 2.0 * (yv / max(Hv - 1, 1)) - 1.0
        else:
            x_norm = (2.0 * (xv + 0.5) / Wv) - 1.0
            y_norm = (2.0 * (yv + 0.5) / Hv) - 1.0

        base = torch.stack([x_norm, y_norm], dim=-1).view(-1, 2)  # [H*W,2]
        return base

    def _get_rect_idx(self, H_eff, W_eff, W_pad, device):
        key = (H_eff, W_eff, W_pad, device.type, device.index)
        t = self._rect_cache.get(key, None)
        if t is None:
            t = _top_left_rect_flat_indices(H_eff, W_eff, W_pad, device=device)  # [N_eff]
            self._rect_cache[key] = t
        return t

    def _get_base(self, H_eff, W_eff, Hv, Wv, device):
        key = (H_eff, W_eff, Hv, Wv, self.align_corners, device.type, device.index)
        t = self._base_cache.get(key, None)
        if t is None:
            t = self._make_base_grid_norm(H_eff, W_eff, Hv, Wv, device=device)  # [N_eff,2]
            self._base_cache[key] = t
        return t

    # 对每个“有效的 src token”，在 vggt 的 37×37 特征图上做 K 点 deformable 采样 + 加权聚合，得到一个几何增强特征 agg，再门控融合回 src，padding 部分保持不变。
    def forward(
        self,
        src_flatten: torch.Tensor,    # [N,1600,256]
        vggt_features: torch.Tensor,  # [N,1369,2048]
        mask_flatten: torch.Tensor,   # [N,1600] True=padding
        src_hw_pad=(40, 40),
        vggt_hw=(37, 37),
    ):
        N, N_src, C_src = src_flatten.shape
        device = src_flatten.device
        H_pad, W_pad = src_hw_pad
        Hv, Wv = vggt_hw

        assert N_src == H_pad * W_pad and C_src == self.d_src
        assert vggt_features.shape[0] == N and vggt_features.shape[1] == Hv * Wv and vggt_features.shape[2] == self.d_vggt
        assert mask_flatten.shape == (N, N_src)

        # 0) 输出初始化（padding 默认保持原样）
        out = src_flatten.clone()

        # 1) 统一把 vggt 投影并 reshape 成 feature map：[N, d_model, Hv, Wv]
        v_map = self.vggt_proj(vggt_features)                         # [N, Hv*Wv, d_model]
        v_map = v_map.view(N, Hv, Wv, self.d_model).permute(0, 3, 1, 2).contiguous()  # [N,d_model,Hv,Wv]

        # 2) 先算每个样本的 (H_eff, W_eff) 并按组收集
        #    这里仍有一个轻量循环（只做 mask -> H/W 推断），重计算都被移到组内 batch 了
        groups = {}  # (H_eff,W_eff) -> list of n
        pad_bool = mask_flatten
        if pad_bool.dtype != torch.bool:
            pad_bool = pad_bool.bool()

        for n in range(N):
            H_eff, W_eff = _infer_effective_hw_from_pad_mask(pad_bool[n], H_pad, W_pad)
            if H_eff == 0 or W_eff == 0:
                continue
            groups.setdefault((H_eff, W_eff), []).append(n)

        # 3) 分组 batch 跑
        if self.align_corners:
            dx_scale = 2.0 / max(Wv - 1, 1)
            dy_scale = 2.0 / max(Hv - 1, 1)
        else:
            dx_scale = 2.0 / Wv
            dy_scale = 2.0 / Hv

        for (H_eff, W_eff), idx_list in groups.items():
            g = torch.tensor(idx_list, device=device, dtype=torch.long)   # [Ng]
            rect_idx = self._get_rect_idx(H_eff, W_eff, W_pad, device)    # [N_eff]
            base = self._get_base(H_eff, W_eff, Hv, Wv, device)           # [N_eff,2]
            N_eff = rect_idx.numel()
            Ng = g.numel()

            # src_q: [Ng, N_eff, d_src]
            src_q = src_flatten.index_select(0, g).index_select(1, rect_idx)
            src_qn = self.q_norm(src_q)  # LN supports [...,C]

            # flatten to run MLP once
            flat_qn = src_qn.reshape(Ng * N_eff, self.d_src)

            offsets_raw = self.offset_mlp(flat_qn).view(Ng, N_eff, self.K, 2)  # [Ng,N_eff,K,2]
            offsets = torch.tanh(offsets_raw) * self.offset_scale

            dx = offsets[..., 0] * dx_scale
            dy = offsets[..., 1] * dy_scale

            # grid: [Ng, N_eff, K, 2]
            # base: [N_eff,2] -> [Ng,N_eff,K,2]
            grid = base[None, :, None, :].expand(Ng, N_eff, self.K, 2).clone()
            grid[..., 0] += dx
            grid[..., 1] += dy
            grid.clamp_(-1.0, 1.0)

            # sampled: [Ng, d_model, N_eff, K]
            sampled = F.grid_sample(
                v_map.index_select(0, g),  # [Ng,d_model,Hv,Wv]
                grid,                      # [Ng,N_eff,K,2]
                mode="bilinear",
                padding_mode="zeros",
                align_corners=self.align_corners,
            )

            # weights: [Ng,N_eff,K] -> [Ng,1,N_eff,K]
            w = self.weight_mlp(flat_qn).view(Ng, N_eff, self.K)
            w = F.softmax(w, dim=-1).unsqueeze(1)

            # agg: [Ng,d_model,N_eff]
            agg = (sampled * w).sum(dim=-1)  # sum over K
            # -> [Ng,N_eff,d_model]
            agg = agg.permute(0, 2, 1).contiguous()

            # gate + fuse (完全一致的公式)
            gate = self.fuse_gate(torch.cat([src_q, agg], dim=-1))  # [Ng,N_eff,d_model]
            fused = src_q + self.out_proj(gate * agg)               # [Ng,N_eff,d_src]

            # 写回（对这个组一次性写）
            out.index_select(0, g)  # 只是为了语义清楚，不用也行
            out[g[:, None], rect_idx[None, :], :] = fused

        return out

class ViewEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0, norm_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,  # x: [B, S, C]
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def _sa_block(self, x, key_padding_mask, return_attn):
        # attn_w: [B, heads, S, S] if return_attn=True else None
        attn_out, attn_w = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,     # [B,S], True means ignore
            need_weights=return_attn,
            average_attn_weights=False,
        )
        return self.dropout1(attn_out), attn_w

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.act(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, x, key_padding_mask=None, return_attn=False):
        attn_w = None
        if self.norm_first:
            sa, attn_w = self._sa_block(self.norm1(x), key_padding_mask, return_attn)
            x = x + sa
            x = x + self._ff_block(self.norm2(x))
        else:
            sa, attn_w = self._sa_block(x, key_padding_mask, return_attn)
            x = self.norm1(x + sa)
            x = self.norm2(x + self._ff_block(x))
        return x, attn_w


class ViewTransformerFuser(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.0, norm_first=True):
        super().__init__()
        self.layers = nn.ModuleList([
            ViewEncoderLayerWithAttn(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=norm_first
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None, return_attn=False):
        """
        x: [B, S, C]  (你这里 B=Nq)
        key_padding_mask: [B, S] bool
        return_attn: if True returns (x, attn_list), else returns x
        """
        attn_list = [] if return_attn else None

        for layer in self.layers:
            x, attn_w = layer(x, key_padding_mask=key_padding_mask, return_attn=return_attn)
            if return_attn:
                attn_list.append(attn_w)  # [B, heads, S, S]

        x = self.norm(x)
        if return_attn:
            return x, attn_list
        return x


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
        cycle_pix_thr=40.0,
        embed_dim=256,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rpe_type = rpe_type
        self.feature_stride = feature_stride
        self.cycle_pix_thr = cycle_pix_thr
        self.embed_dim = embed_dim
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

        # MVF config
        # ---- view-fuse hyperparams (put into config if you like) ----
        self.view_fuse_heads  = getattr(self, "view_fuse_heads", 8)
        # self.view_fuse_heads  = getattr(self, "view_fuse_heads", 4)
        self.view_fuse_layers = getattr(self, "view_fuse_layers", 2)
        self.view_fuse_ffn_mult = getattr(self, "view_fuse_ffn_mult", 4)
        self.view_fuse_drop = getattr(self, "view_fuse_drop", 0.0)

        # optional: run view-fuse in fp32 for stability
        self.view_fuse_fp32 = getattr(self, "view_fuse_fp32", False)

        assert self.embed_dim % self.view_fuse_heads == 0, f"C={self.embed_dim} must be divisible by heads={self.view_fuse_heads}"

        # enc_layer = nn.TransformerEncoderLayer(
        #     d_model=self.embed_dim,
        #     nhead=self.view_fuse_heads,
        #     dim_feedforward=self.view_fuse_ffn_mult * self.embed_dim,
        #     dropout=self.view_fuse_drop,
        #     activation="gelu",
        #     batch_first=True,
        #     norm_first=True,
        # )
        # self.view_fuser = nn.TransformerEncoder(enc_layer, num_layers=self.view_fuse_layers)
        # 它把“每个 query 在 S 个 view 上的特征序列”当成一句话，用 Transformer 在 view 之间做信息交换；无效 view 用 key_padding_mask=True 屏蔽掉。
        self.view_fuser = ViewTransformerFuser(
            d_model=self.embed_dim,
            nhead=self.view_fuse_heads,
            num_layers=self.view_fuse_layers,
            dim_feedforward=self.view_fuse_ffn_mult * self.embed_dim,
            dropout=self.view_fuse_drop,
            norm_first=True,
        )
        
        self.view_fuse_norm = nn.LayerNorm(self.embed_dim)
        
    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        cpb_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, out_dim, bias=False))
        return cpb_mlp

    def tid_to_xy_center(self, tid, Wg_v, Hg_v, W_img, H_img):
        
        tid = tid.long().clamp(0, Hg_v * Wg_v - 1)  # 先保证 tid 合法
        tx = (tid % Wg_v)
        ty = (tid // Wg_v)

        xn = ((tx.float() + 0.5) / float(Wg_v)).clamp(0.0, 1.0)
        yn = ((ty.float() + 0.5) / float(Hg_v)).clamp(0.0, 1.0)

        x_c = xn * float(W_img - 1)
        y_c = yn * float(H_img - 1)
        return torch.stack([x_c, y_c], dim=1)
    
    def _vggt_map_centers_onepair(self, vggt_b, s_src, s_tgt, cxcy_src, cycle_pix_thr, Wg_v, Hg_v, W_img, H_img, device):
        # vggt_b: [S,1369,Cv], cxcy_src: [Np,2] pixels
        if s_tgt == s_src:
            ok = torch.ones((cxcy_src.shape[0],), device=device, dtype=torch.bool)
            return ok, cxcy_src

        src_feat = F.normalize(vggt_b[s_src], dim=-1, eps=1e-8)
        tgt_feat = F.normalize(vggt_b[s_tgt], dim=-1, eps=1e-8)

        xy = cxcy_src.float()
        x = xy[:, 0].clamp(0.0, float(W_img - 1))
        y = xy[:, 1].clamp(0.0, float(H_img - 1))
        xn = x / max(float(W_img - 1), 1.0)
        yn = y / max(float(H_img - 1), 1.0)

        px = torch.floor(xn * Wg_v).long().clamp(0, Wg_v - 1)
        py = torch.floor(yn * Hg_v).long().clamp(0, Hg_v - 1)
        src_tid = (py * Wg_v + px).long()  # [Np]

        qv = src_feat.index_select(0, src_tid)     # [Np,Cv]
        sim_st = qv @ tgt_feat.t()                 # [Np,1369]
        _, j = sim_st.max(dim=1)                   # [Np]

        tv = tgt_feat.index_select(0, j)           # [Np,Cv]
        sim_ts = tv @ src_feat.t()                 # [Np,1369]
        back_tid = sim_ts.argmax(dim=1)            # [Np]



        cxcy_tgt  = self.tid_to_xy_center(j, Wg_v, Hg_v, W_img, H_img)
        cxcy_back = self.tid_to_xy_center(back_tid, Wg_v, Hg_v, W_img, H_img)

        cycle_dist = torch.linalg.norm(cxcy_back - xy, dim=1)
        ok = cycle_dist <= cycle_pix_thr

        # keep stable center for invalid ones
        cxcy_tgt = torch.where(ok[:, None], cxcy_tgt, cxcy_src)
        return ok, cxcy_tgt
    
    
    def _build_rpe_from_cxcywh_batched(self, ref_cxcywh, pos_x, pos_y):
        """
        ref_cxcywh: [M, Np, 4]  (pixels, cxcywh)
        pos_x: [1,1,w,1] float32 * stride
        pos_y: [1,1,h,1] float32 * stride
        return: rpe [M, H, Np, HW]
        """
        M, Np, _ = ref_cxcywh.shape
        cxcy = ref_cxcywh[..., 0:2]
        wh   = ref_cxcywh[..., 2:4]
        xy0  = cxcy - 0.5 * wh
        xy1  = cxcy + 0.5 * wh
        ref_xyxy = torch.cat([xy0, xy1], dim=-1)[:, :, None, :]  # [M,Np,1,4]

        if self.rpe_type == "abs_log8":
            delta_x = ref_xyxy[..., 0::2] - pos_x  # [M,Np,w,2]
            delta_y = ref_xyxy[..., 1::2] - pos_y  # [M,Np,h,2]
            delta_x = torch.sign(delta_x) * torch.log2(torch.abs(delta_x) + 1.0) / np.log2(8)
            delta_y = torch.sign(delta_y) * torch.log2(torch.abs(delta_y) + 1.0) / np.log2(8)
        elif self.rpe_type == "linear":
            delta_x = ref_xyxy[..., 0::2] - pos_x
            delta_y = ref_xyxy[..., 1::2] - pos_y
        else:
            raise NotImplementedError

        # cpb_mlp: (...,2) -> (...,H)
        rpe_x = self.cpb_mlp1(delta_x)  # [M,Np,w,H]
        rpe_y = self.cpb_mlp2(delta_y)  # [M,Np,h,H]
        rpe = (rpe_x[:, :, None] + rpe_y[:, :, :, None]).flatten(2, 3)   # [M,Np,HW,H]
        rpe = rpe.permute(0, 3, 1, 2).contiguous()                        # [M,H,Np,HW]
        return rpe

    # mvf-batch
    def forward(
        self,
        vggt_features,          # [B, S, 1369, Cv]
        query,                  # [B*S, Nq, C]
        reference_2d,           # [B*S, Np, 1, 4] (cxcywh in pixels)
        k_input_flatten,        # [B*S, HW, C]
        v_input_flatten,        # [B*S, HW, C]
        input_spatial_shapes,   # [[h,w]] single-scale
        input_padding_mask=None,# [B*S, HW] (0/1 or bool)
        box_attn_prior_mask=None,# [Nq] bool OR long indices,
        vHW_patch=(16, 16)
    ):
        assert input_spatial_shapes.size(0) == 1, "single-scale only"
        h_t, w_t = input_spatial_shapes[0]
        h = int(h_t.item()) if torch.is_tensor(h_t) else int(h_t)
        w = int(w_t.item()) if torch.is_tensor(w_t) else int(w_t)
        stride = float(self.feature_stride)

        # ---- infer B,S ----
        assert vggt_features.ndim == 4, f"vggt_features should be [B,S,1369,Cv], got {tuple(vggt_features.shape)}"
        B_v, S, Nt_v, Cv = vggt_features.shape
        assert Nt_v == vHW_patch[0] * vHW_patch[1], f"expect 1369 vggt tokens, got {Nt_v}"

        BS, HW, C = k_input_flatten.shape
        assert BS % S == 0, f"BS={BS} must be divisible by S={S}"
        B = BS // S

        BS2, Nq, C2 = query.shape
        assert BS2 == BS and C2 == C

        # ---- ref ----
        assert reference_2d.ndim == 4 and reference_2d.shape[0] == BS and reference_2d.shape[2] == 1 and reference_2d.shape[3] == 4
        ref = reference_2d.squeeze(2)      # [BS,Np,4]
        Np = ref.shape[1]
        ref = ref.view(B, S, Np, 4)        # [B,S,Np,4]
        assert Nq >= Np, f"Nq={Nq} must be >= Np={Np}"

        device = query.device
        dtype  = query.dtype

        if box_attn_prior_mask is None:
            raise ValueError("box_attn_prior_mask must be provided (it selects proposal queries that use RPE).")
        assert box_attn_prior_mask.ndim == 1 and box_attn_prior_mask.numel() == Nq
        box_attn_prior_mask = box_attn_prior_mask.to(device=device)

        # prop indices (长度 Np)
        if box_attn_prior_mask.dtype == torch.bool:
            prop_idx = torch.nonzero(box_attn_prior_mask, as_tuple=False).squeeze(1)  # [Np]
            assert prop_idx.numel() == Np, f"box_attn_prior_mask.sum() must equal Np={Np}, got {prop_idx.numel()}"
        else:
            prop_idx = box_attn_prior_mask.long()
            assert prop_idx.numel() == Np, f"box_attn_prior_mask must provide Np={Np} indices, got {prop_idx.numel()}"

        # ---- reshape to [B,S,...] ----
        query_ = query.view(B, S, Nq, C)              # [B,S,Nq,C]
        k_in   = k_input_flatten.view(B, S, HW, C)    # [B,S,HW,C]
        v_in   = v_input_flatten.view(B, S, HW, C)    # [B,S,HW,C]

        # ---- project q/k/v ----
        k = self.k(k_in).reshape(B, S, HW, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()  # [B,S,H,HW,Dh]
        v = self.v(v_in).reshape(B, S, HW, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()  # [B,S,H,HW,Dh]
        q = self.q(query_).reshape(B, S, Nq, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous() # [B,S,H,Nq,Dh]
        q = q * self.scale

        # ---- cache pos_x / pos_y (float32) ----
        if not hasattr(self, "_pos_cache"):
            self._pos_cache = {}
        pos_key = (device, h, w, float(stride))
        if pos_key in self._pos_cache:
            pos_x, pos_y = self._pos_cache[pos_key]
        else:
            pos_x = torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device)[None, None, :, None] * float(stride)  # [1,1,w,1]
            pos_y = torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device)[None, None, :, None] * float(stride)  # [1,1,h,1]
            self._pos_cache[pos_key] = (pos_x, pos_y)

        # ---- cubify mapping settings ----
        H_img, W_img = int(round(h * stride)), int(round(w * stride))
        # ---- vggt mapping settings ----
        Hg_v, Wg_v = vHW_patch[0], vHW_patch[1]

        # ============================================================
        # 1) VGGT cycle mapping: for ALL (b, s_src, s_tgt, proposal)
        # ============================================================
        # normalize vggt: [B,S,Nv,Cv]
        vggt_n = F.normalize(vggt_features, dim=-1, eps=1e-8)

        # src cxcy, wh
        cxcy_src = ref[..., :2]  # [B,S,Np,2]
        wh_src   = ref[..., 2:]  # [B,S,Np,2]

        # compute src_tid [B,S,Np]
        x = cxcy_src[..., 0].clamp(0.0, float(W_img - 1))
        y = cxcy_src[..., 1].clamp(0.0, float(H_img - 1))
        xn = x / max(float(W_img - 1), 1.0)
        yn = y / max(float(H_img - 1), 1.0)
        px = torch.floor(xn * Wg_v).long().clamp(0, Wg_v - 1)
        py = torch.floor(yn * Hg_v).long().clamp(0, Hg_v - 1)
        src_tid = (py * Wg_v + px).long()  # [B,S,Np]

        # qv: src_feat at src_tid -> [B,S,Np,Cv]
        # take_along_dim needs index shape [B,S,Np,1] on dim=2
        qv = torch.take_along_dim(vggt_n, src_tid[..., None].expand(B, S, Np, Cv), dim=2)  # [B,S,Np,Cv]

        # sim_st: (qv) dot (tgt_feat) for all tgt views
        # qv: [B,S,Np,Cv], vggt_n: [B,T,Nv,Cv]
        # -> [B,S,Np,T,Nv]
        sim_st = torch.einsum("bspc,btvc->bsptv", qv, vggt_n)  # [B,S,Np,S,Nv]
        j = sim_st.argmax(dim=-1)                               # [B,S,Np,S]  (tgt token id)

        # expand vggt to [B, S_src, Np, S_tgt, Nv, Cv]
        vggt_exp = vggt_n[:, None, None, :, :, :].expand(B, S, Np, S, Nt_v, Cv)

        # build gather indices for Nv-dim (dim=4): [B, S_src, Np, S_tgt, 1, Cv]
        idx = j[..., None, None].expand(B, S, Np, S, 1, Cv)

        # gather along Nv dimension
        tv = torch.gather(vggt_exp, dim=4, index=idx).squeeze(4)  # [B, S_src, Np, S_tgt, Cv]

        # sim_ts: tv dot src_feat -> [B,S,Np,S,Nv]
        sim_ts = torch.einsum("bsptc,bsvc->bsptv", tv, vggt_n)  # [B,S,Np,S,Nv]
        back_tid = sim_ts.argmax(dim=-1)                        # [B,S,Np,S]

        # convert ids to pixel centers
        # tid_to_xy_center expects [*,] and returns [*,2]
        cxcy_tgt  = self.tid_to_xy_center(j.reshape(-1), Wg_v, Hg_v, W_img, H_img).view(B, S, Np, S, 2)
        cxcy_back = self.tid_to_xy_center(back_tid.reshape(-1), Wg_v, Hg_v, W_img, H_img).view(B, S, Np, S, 2)

        # cycle check
        xy = cxcy_src[:, :, :, None, :]  # [B,S,Np,1,2]
        cycle_dist = torch.linalg.norm(cxcy_back - xy, dim=-1)   # [B,S,Np,S]
        ok = cycle_dist <= float(self.cycle_pix_thr)             # [B,S,Np,S] bool

        # self-view always ok, and cxcy_tgt = cxcy_src
        eye = torch.eye(S, device=device, dtype=torch.bool)[None, None, None, :, :]  # [1,1,1,S,S]
        # expand to [B,S,Np,S,S] then pick diagonal along last axis? easier:
        # we want mask_self[b, s_src, p, s_tgt] == (s_tgt == s_src)
        s_src_ids = torch.arange(S, device=device)[None, :, None, None]  # [1,S,1,1]
        s_tgt_ids = torch.arange(S, device=device)[None, None, None, :]  # [1,1,1,S]
        self_mask = (s_tgt_ids == s_src_ids)  # [1,S,1,S] broadcast to [B,S,Np,S]
        self_mask = self_mask.expand(B, S, Np, S)

        ok = torch.where(self_mask, torch.ones_like(ok), ok)
        cxcy_tgt = torch.where(self_mask[..., None], xy.expand(B, S, Np, S, 2), cxcy_tgt)

        # invalid -> keep src center stable
        cxcy_tgt = torch.where(ok[..., None], cxcy_tgt, xy.expand(B, S, Np, S, 2))

        # build ref_tgt: [B,S,Np,S,4] then reshape to M=B*S*S
        ref_tgt = torch.cat([cxcy_tgt, wh_src[:, :, :, None, :].expand(B, S, Np, S, 2)], dim=-1)  # [B,S,Np,S,4]
        # permute to [B,S,S,Np,4]
        ref_tgt = ref_tgt.permute(0, 1, 3, 2, 4).contiguous()  # [B,S,S,Np,4]
        ok_all  = ok.permute(0, 1, 3, 2).contiguous()          # [B,S,S,Np]  (for vmask)

        # ============================================================
        # 2) base attention for ALL (src,tgt) + add RPE (scatter_add)
        # ============================================================
        # attn_all: [B, S_src, S_tgt, H, Nq, HW]
        attn_all = torch.einsum("bshnd,bthmd->bsthnm", q, k)  # [B,S,S,H,Nq,HW]

        # RPE: compute for all pairs (b, s_src, s_tgt) in one go
        M = B * S * S
        ref_tgt_M = ref_tgt.view(M, Np, 4)
        rpe_M = self._build_rpe_from_cxcywh_batched(ref_tgt_M, pos_x, pos_y)  # [M,H,Np,HW]
        # gate invalid proposals
        ok_M = ok_all.view(M, Np).to(rpe_M.dtype)  # [M,Np]
        rpe_M = rpe_M * ok_M[:, None, :, None]     # [M,H,Np,HW]
        rpe_all = rpe_M.view(B, S, S, self.num_heads, Np, HW)  # [B,S,S,H,Np,HW]

        # scatter_add RPE into proposal query positions along Nq-dim (dim=4)
        idx = prop_idx.view(1, 1, 1, 1, Np, 1).expand(B, S, S, self.num_heads, Np, HW)
        attn_all = attn_all.scatter_add(dim=4, index=idx, src=rpe_all)

        # padding mask (per tgt view)
        if input_padding_mask is not None:
            pad = input_padding_mask.view(B, S, HW).to(device=device)
            if pad.dtype == torch.bool:
                pad = pad.to(attn_all.dtype)
            else:
                pad = pad.to(attn_all.dtype)
            # broadcast to [B,1,S,1,1,HW] then add
            attn_all = attn_all + pad[:, None, :, None, None, :] * (-100.0)

        # softmax + dropout
        fmin, fmax = torch.finfo(attn_all.dtype).min, torch.finfo(attn_all.dtype).max
        attn_all = torch.clamp(attn_all, min=fmin, max=fmax)
        attn_all = self.softmax(attn_all)
        attn_all = self.attn_drop(attn_all)

        # aggregate values: [B,S,S,H,Nq,Dh] -> [B,S,S,Nq,C]
        x_all = torch.einsum("bsthnm,bthmd->bsthnd", attn_all, v)                  # [B,S,S,H,Nq,Dh]
        x_all = x_all.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, S, S, Nq, C)  # [B,S,S,Nq,C]

        # ============================================================
        # 3) view fuse: flatten (B*S*Nq) as batch, S as sequence length
        # ============================================================
        # x_view: [B,S_src,Nq,S_tgt,C]
        x_view = x_all.permute(0, 1, 3, 2, 4).contiguous()  # [B,S,Nq,S,C]

        # vmask: [B,S,Nq,S]
        vmask = torch.ones((B, S, Nq, S), device=device, dtype=x_view.dtype)
        ok_float = ok_all.to(x_view.dtype)  # [B,S,S,Np]
        # scatter ok into proposal query positions (dim=2)
        idx2 = prop_idx.view(1, 1, Np, 1).expand(B, S, Np, S)
        vmask = vmask.scatter(dim=2, index=idx2, src=ok_float.permute(0,1,3,2))  # ok -> [B,S,Np,S]
        # self-view always valid
        vmask[:, torch.arange(S, device=device), :, torch.arange(S, device=device)] = 1.0

        x_view = x_view * vmask[..., None]
        key_pad = (vmask <= 0.5)  # [B,S,Nq,S] True=ignore

        x_in = x_view.view(B * S * Nq, S, C)
        key_pad_in = key_pad.view(B * S * Nq, S)

        if self.view_fuse_fp32 and x_in.dtype in (torch.float16, torch.bfloat16):
            x_in_f = x_in.float()
        else:
            x_in_f = x_in

        x_fused = self.view_fuser(x_in_f, key_padding_mask=key_pad_in)  # [B*S*Nq, S, C]

        if x_fused.dtype != x_in.dtype:
            x_fused = x_fused.to(x_in.dtype)

        x_fused = x_fused.view(B, S, Nq, S, C)

        # take src position (for each src view, pick seq index = s_src)
        gather_idx = torch.arange(S, device=device).view(1, S, 1, 1, 1).expand(B, S, Nq, 1, 1)
        x_avg = x_fused.gather(dim=3, index=gather_idx.expand(B, S, Nq, 1, C)).squeeze(3)  # [B,S,Nq,C]

        # norm + proj
        x_avg = self.view_fuse_norm(x_avg)
        x_avg = self.proj(x_avg)
        x_avg = self.proj_drop(x_avg)

        out = x_avg.view(B * S, Nq, C)
        return out


# # Plain-DETR.
# class GlobalCrossAttention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads,
#         qkv_bias=True,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         rpe_hidden_dim=512,
#         rpe_type='linear',
#         feature_stride=16,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.rpe_type = rpe_type
#         self.feature_stride = feature_stride

#         # FFN.
#         self.cpb_mlp1 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)
#         self.cpb_mlp2 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.softmax = nn.Softmax(dim=-1)

#     def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
#         cpb_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=True),
#                                 nn.ReLU(inplace=True),
#                                 nn.Linear(hidden_dim, out_dim, bias=False))
#         return cpb_mlp

#     def forward(
#         self,
#         vggt_features,
#         query,
#         reference_2d,
#         k_input_flatten,
#         v_input_flatten,
#         input_spatial_shapes,
#         input_padding_mask=None,
#         box_attn_prior_mask=None
#     ):
#         assert input_spatial_shapes.size(0) == 1, 'This is designed for single-scale decoder.'
#         h, w = input_spatial_shapes[0]
#         stride = self.feature_stride

#         ref_2d_xyxy = torch.cat([
#             reference_2d[:, :, :, :2] - reference_2d[:, :, :, 2:] / 2,
#             reference_2d[:, :, :, :2] + reference_2d[:, :, :, 2:] / 2,
#         ], dim=-1)  # B, nQ, 1, 4

#         pos_x = torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=w.device)[None, None, :, None] * stride  # 1, 1, w, 1
#         pos_y = torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=h.device)[None, None, :, None] * stride  # 1, 1, h, 1

#         if self.rpe_type == 'abs_log8':
#             delta_x = ref_2d_xyxy[..., 0::2] - pos_x  # B, nQ, w, 2
#             delta_y = ref_2d_xyxy[..., 1::2] - pos_y  # B, nQ, h, 2
#             delta_x = torch.sign(delta_x) * torch.log2(torch.abs(delta_x) + 1.0) / np.log2(8)
#             delta_y = torch.sign(delta_y) * torch.log2(torch.abs(delta_y) + 1.0) / np.log2(8)
#         elif self.rpe_type == 'linear':
#             delta_x = ref_2d_xyxy[..., 0::2] - pos_x  # B, nQ, w, 2
#             delta_y = ref_2d_xyxy[..., 1::2] - pos_y  # B, nQ, h, 2
#         else:
#             raise NotImplementedError
#         rpe_x, rpe_y = self.cpb_mlp1(delta_x), self.cpb_mlp2(delta_y)  # B, nQ, w/h, nheads
#         rpe = (rpe_x[:, :, None] + rpe_y[:, :, :, None]).flatten(2, 3) # B, nQ, h, w, nheads ->  B, nQ, h*w, nheads
#         rpe = rpe.permute(0, 3, 1, 2)

#         B_, N, C = k_input_flatten.shape
#         k = self.k(k_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         v = self.v(v_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         B_, N, C = query.shape
#         q = self.q(query).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         q = q * self.scale

#         attn = q @ k.transpose(-2, -1)

#         # Only apply the RPE if the prior mask informs us to. Certain queries (like metric tokens) do not need priors.
#         if box_attn_prior_mask is not None:
#             attn[:, :, box_attn_prior_mask] += rpe
#         else:
#             attn += rpe

#         if input_padding_mask is not None:
#             attn += input_padding_mask[:, None, None] * -100

#         fmin, fmax = torch.finfo(attn.dtype).min, torch.finfo(attn.dtype).max
#         torch.clip_(attn, min=fmin, max=fmax)

#         attn = self.softmax(attn)
#         attn = self.attn_drop(attn)

#         x = attn @ v

#         x = x.transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x

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
        self.norm2 = nn.LayerNorm(d_model) #保证训练稳定、防止梯度爆炸

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

    '''
    changed by lyq to make queries interact with each other
    '''
    def forward(
        self,
        vggt_features,
        tgt,
        query_pos,
        reference_2d,
        src,
        src_pos_embed,
        src_spatial_shapes,
        batch_img,
        src_padding_mask=None,
        self_attn_mask=None,
        box_attn_prior_mask=None,
        vHW_patch=(16, 16)
    ):
        N_batch, N_img = batch_img
        # self_attn_mask = self_attn_mask[:2].flatten()
        Q = self_attn_mask.size(0)
        # big_mask = torch.zeros(2*N, 2*N, dtype=torch.bool).to(self_attn_mask.device)  # 默认全 True=屏蔽
        # big_mask[:N, :N] = self_attn_mask # 图1 的区域
        # big_mask[N:, N:] = self_attn_mask.clone() # 图2 的区域
        # self_attn_mask = big_mask
        A_block = self_attn_mask.expand(N_img, N_img, Q, Q)
        A_block = A_block.permute(0, 2, 1, 3).reshape(N_img*Q, N_img*Q)
        A_mask = torch.block_diag(*([A_block] * N_batch))
        off_diag = ~torch.block_diag(*([torch.ones_like(A_block)] * N_batch))
        self_attn_mask = off_diag | A_mask
        
        # self attention
        # 1️⃣ 自注意力：query 之间交互 让所有 queries（对象 tokens）互相沟通，理解彼此的上下文关系（比如不同物体之间的相对位置/类别信息）
        tgt2 = self.norm2(tgt)
        Ba, Nq, Nc = tgt2.shape
        tgt2_merged = tgt2.reshape(1, Ba * Nq, Nc)
        q = k = self.with_pos_embed(tgt2_merged, query_pos.reshape(1, Ba * Nq, Nc))
        # nn.MultiheadAttention 的原始输入是 [seq_len, batch, embed_dim]
        tgt2_merged = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt2_merged.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt2 = tgt2_merged.reshape(Ba, Nq, Nc)
        tgt = tgt + self.dropout2(tgt2)

        # 2️⃣ 全局跨注意力：与图像特征交互
        # global cross attention
        # # 让每个 query 从 encoder 的图像特征（memory）中读取与自己位置相关的特征；这里使用 GlobalCrossAttention，支持相对位置偏置 RPE（Relative Positional Encoding），并通过 reference_2d（每个 query 对应的 2D box）指导注意力范围
        if self.xattn is not None:
            tgt2 = self.norm1(tgt)
            tgt2 = self.xattn(
                vggt_features,
                self.with_pos_embed(tgt2, query_pos),
                reference_2d,
                self.with_pos_embed(src, src_pos_embed),
                src,
                src_spatial_shapes,
                src_padding_mask,
                box_attn_prior_mask=box_attn_prior_mask,
                vHW_patch=(vHW_patch[0], vHW_patch[1]) #TODO:
            )

            tgt = tgt + self.dropout1(tgt2)

        # ffn 3️⃣ FFN 前馈网络 对每个 token 做非线性变换，增强特征表达能力
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)

        return tgt
    # '''
    # original
    # '''
    # def forward(
    #     self,
    #     tgt,
    #     query_pos,
    #     reference_2d,
    #     src,
    #     src_pos_embed,
    #     src_spatial_shapes,
    #     src_padding_mask=None,
    #     self_attn_mask=None,
    #     box_attn_prior_mask=None
    # ):
        
        
    #     # self attention
    #     # 1️⃣ 自注意力：query 之间交互 让所有 queries（对象 tokens）互相沟通，理解彼此的上下文关系（比如不同物体之间的相对位置/类别信息）
    #     tgt2 = self.norm2(tgt)

    #     q = k = self.with_pos_embed(tgt2, query_pos)
    #     # nn.MultiheadAttention 的原始输入是 [seq_len, batch, embed_dim]
    #     tgt2 = self.self_attn(
    #         q.transpose(0, 1),
    #         k.transpose(0, 1),
    #         tgt2.transpose(0, 1),
    #         attn_mask=self_attn_mask,
    #     )[0].transpose(0, 1)
    #     tgt = tgt + self.dropout2(tgt2)

    #     # 2️⃣ 全局跨注意力：与图像特征交互
    #     # global cross attention
    #     # # 让每个 query 从 encoder 的图像特征（memory）中读取与自己位置相关的特征；这里使用 GlobalCrossAttention，支持相对位置偏置 RPE（Relative Positional Encoding），并通过 reference_2d（每个 query 对应的 2D box）指导注意力范围
    #     if self.xattn is not None:
    #         tgt2 = self.norm1(tgt)
    #         tgt2 = self.xattn(
    #             self.with_pos_embed(tgt2, query_pos), #q
    #             reference_2d,
    #             self.with_pos_embed(src, src_pos_embed), #k
    #             src, #v
    #             src_spatial_shapes,
    #             src_padding_mask,
    #             box_attn_prior_mask=box_attn_prior_mask
    #         )

    #         tgt = tgt + self.dropout1(tgt2)

    #     # ffn 3️⃣ FFN 前馈网络 对每个 token 做非线性变换，增强特征表达能力
    #     tgt2 = self.norm3(tgt)
    #     tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
    #     tgt = tgt + self.dropout4(tgt2)

    #     return tgt

# This should be super vague, and take in "prompts" as queries and simply run them through
# the underlying transformer.
class PromptDecoder(nn.Module):
    def __init__(
            self, embed_dim, layer, num_layers, predictors, norm=nn.Identity()):
        super(PromptDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers

        self.predictors = nn.ModuleList([nn.ModuleList([copy.deepcopy(p) for p in predictors]) for _ in range(num_layers)])
        self.norm = norm

    def forward(
        self, vggt_features, src, src_pos_embed, src_padding_mask, src_spatial_shapes, level_start_index, valid_ratios,
        prompts, sensor, batch, img_num, vHW_patch):
        # Not the best assumption for now, but assume we can treat the "prompt" in a uniform manner as "reference.
        # Interleave so we have List[List[Instances]] of size # of instances x # of prompts
        # Original implementation:
        reference = [Instances3D.cat([prompt.pred[idx] for prompt in prompts if prompt.box_attn_prior_mask.any()]) for idx in range(prompts[0].batch_size)] #batch size: N
        
        N_batch = prompts[0].batch_size
        N_emb = src.shape[-1]
        
        
        #cat all prompts of a batch
        prompts = Prompt.cat(prompts)        
        output = prompts.query
        intermediate = []
        intermediate_preds = []

        # TODO:reshape回去
        src=src.reshape(N_batch,-1,N_emb)
        src_pos_embed=src_pos_embed.reshape(N_batch,-1,N_emb)
        src_padding_mask=src_padding_mask.reshape(N_batch,-1)

        image_size = sensor["image"].data.image_sizes[0]

        for lid, layer in enumerate(self.layers): #6
            # Currently, the only thing that influences the RPE is the current 2D box prediction.
            reference_2d = torch.stack([ref_.pred_boxes for ref_ in reference]).detach()
            output = layer(
                vggt_features,
                output,
                prompts.pos_embed,
                # For now, we only bias the attention based on the current 2D box. 
                reference_2d[:, :, None],
                src,
                src_pos_embed,
                src_spatial_shapes,
                (batch, img_num), #TODO: only for query in a batch version
                src_padding_mask,
                prompts.self_attn_mask,
                box_attn_prior_mask=prompts.box_attn_prior_mask,
                vHW_patch=(vHW_patch[0], vHW_patch[1])
            )
            # B, N_queries, C=256
            output_after_norm = self.norm(output)

            # pred_instances = [Instances3D(image_size) for image_size in sensor["image"].data.image_sizes]
            pred_instances = [Instances3D(image_size) for idx in range(N_batch)]
            
            
            for pred_instances_, reference_ in zip(pred_instances, reference):
                # The previous layer's "predictions", are this rounds, "proposals"
                pred_instances_.proposal_boxes = reference_.pred_boxes
                if hasattr(pred_instances_, "pred_proj_xy"):
                    pred_instances_.proposal_proj_xy = reference_.pred_proj_xy
                    pred_instances_.proposal_z_unscaled = reference_.pred_z_unscaled
                    pred_instances_.proposal_dims = reference_.pred_dims
                    pred_instances_.proposal_pose = reference_.pred_pose
            
            # 6层，每一层都有3个predictor head
            for predictor in self.predictors[lid]:
                # Hacky, but let a predictor alter the output.
                output_after_norm = predictor(output_after_norm, pred_instances, sensor)

            for pred_instances_, output_ in zip(pred_instances, output_after_norm):
                pred_instances_.object_desc = output_

            intermediate.append(output_after_norm)
            intermediate_preds.append(pred_instances)
            reference = pred_instances

        # Always return the full sequence of intermediates.
        return torch.stack(intermediate), intermediate_preds

class PromptEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(PromptEncoder, self).__init__()

        self.embed_dim = embed_dim

class Box2DPromptEncoderLearned(PromptEncoder):
    def __init__(self, embed_dim, max_x=1280, max_y=1280, max_w=1280, max_h=1280, discretization_steps=1):
        super(Box2DPromptEncoderLearned, self).__init__(embed_dim)

        self.x = nn.Embedding(max_x * discretization_steps, embed_dim // 4)
        self.y = nn.Embedding(max_y * discretization_steps, embed_dim // 4)
        self.w = nn.Embedding(max_w * discretization_steps, embed_dim // 4)
        self.h = nn.Embedding(max_h * discretization_steps, embed_dim // 4)
        self.discretization_steps = discretization_steps

        self.register_buffer("min_bounds", torch.tensor([0.0, 0.0, 0.0, 0.0]).float())
        self.register_buffer("max_bounds", torch.tensor([max_x - 1, max_y - 1, max_w - 1, max_h - 1]).float())

    # Usually, this should be detached.
    def forward(self, boxes):
        indexes = self.discretization_steps * torch.clamp(boxes, min=self.min_bounds[None, None], max=self.max_bounds[None, None]).int()
        pos = torch.cat((
            self.x(indexes[..., 0]),
            self.y(indexes[..., 1]),
            self.w(indexes[..., 2]),
            self.h(indexes[..., 3]),
        ), dim=-1)

        return pos

class Predictor(nn.Module):
    def __init__(self, embed_dim):
        super(Predictor, self).__init__()

        self.embed_dim = embed_dim

class ScalePredictor(Predictor):
    def __init__(self, embed_dim):
        super(ScalePredictor, self).__init__(embed_dim)

        # For now, just a simple linear layer.
        self.shift = nn.Linear(embed_dim, 1)
        self.scale = nn.Linear(embed_dim, 1)

    def forward(self, x, proposals, sensor): #save in the 'sensor'
        pred_shift = torch.exp(self.shift(x[:, 0:1]))
        pred_scale = torch.exp(self.scale(x[:, 1:2]))

        shift_scale = torch.cat((pred_shift, pred_scale), dim=-1)
        # Probably better to store on "depth", but we don't have that sensor for
        # monocular.
        scale_infos = sensor["depth"].info if "depth" in sensor else sensor["image"].info
        # for i, info_ in enumerate(scale_infos):
        #     info_.pred_parameters = shift_scale[i]
        
        #TODO: save the scale of all batches and images in all info
        for i, info_ in enumerate(scale_infos):
            info_.pred_parameters = shift_scale

        # Slice off.
        return x[:, 2:]        

class ClassPredictor(Predictor):
    def __init__(self, embed_dim, num_classes, prior_prob=0.01, num_layers=None):
        super(ClassPredictor, self).__init__(embed_dim)

        self.num_classes = num_classes

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        if num_layers is None:
            self.linear = nn.Linear(embed_dim, num_classes)
            self.linear.bias.data.fill_(bias_value)            
        else:
            self.linear = MLP(embed_dim, embed_dim, num_classes, num_layers)            
            self.linear.layers[-1].bias.data.fill_(bias_value)

    def forward(self, x, proposals, sensor):
        logits = self.linear(x)

        for proposals_, logits_ in zip(proposals, logits):
            proposals_.pred_logits = logits_

        return x    

class Box2DPredictor(Predictor):
    def __init__(self, embed_dim, num_layers=3):
        super(Box2DPredictor, self).__init__(embed_dim)

        self.mlp = MLP(embed_dim, embed_dim, 4, num_layers)

        nn.init.constant_(self.mlp.layers[-1].weight.data, 0)
        nn.init.constant_(self.mlp.layers[-1].bias.data, 0)

class DeltaBox2DTransform(nn.Module):
    def __init__(self, wh_ratio_clip=0.016, clamp_to_border=True, center_clamp=None,
                 means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
        super(DeltaBox2DTransform, self).__init__()
        
        self._wh_ratio_clip = wh_ratio_clip
        self._clamp_to_border = clamp_to_border
        self._center_clamp = center_clamp

        self.register_buffer("means", torch.tensor(means), False)
        self.register_buffer("stds", torch.tensor(stds), False)

    def get_deltas(self, src_boxes, target_boxes):
        # hack for matcher
        if src_boxes.size() != target_boxes.size():
            src_boxes = src_boxes[:, None]
            target_boxes = target_boxes[None]

        src_boxes = src_boxes.float()
        target_boxes = target_boxes.float()
        px, py, pw, ph = src_boxes.unbind(-1)
        gx, gy, gw, gh = target_boxes.unbind(-1)

        dx = (gx - px) / (pw + 0.1)
        dy = (gy - py) / (ph + 0.1)
        dw = torch.log(gw / (pw + 0.1))
        dh = torch.log(gh / (ph + 0.1))

        deltas = torch.stack([dx, dy, dw, dh], dim=-1)
        deltas = deltas.sub_(self.means[None]).div_(self.stds[None])

        return deltas
    
    def apply_deltas(self, deltas, boxes, clamp_shape=None):
        dxy = deltas[..., :2]
        dwh = deltas[..., 2:]

        # Compute width/height of each roi
        pxy = boxes[..., :2]
        pwh = boxes[..., 2:]

        # dxdy is in terms of wh units.
        dxy_wh = pwh * dxy

        # Don't allow the delta in WH to go larger than this.
        max_ratio = np.abs(np.log(self._wh_ratio_clip))
        if self._center_clamp is not None:
            dxy_wh = torch.clamp(dxy_wh, max=self._center_clamp, min=-self._center_clamp)
            dwh = torch.clamp(dwh, max=max_ratio)
        else:
            dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

        gxy = pxy + dxy_wh
        gwh = pwh * dwh.exp()
        x1y1 = gxy - (gwh * 0.5)
        x2y2 = gxy + (gwh * 0.5)
        bboxes = torch.cat([x1y1, x2y2], dim=-1)

        # Don't allow predictions to leave the image.
        if self._clamp_to_border and (clamp_shape is not None):
            # This code actually clamps.
            bboxes.clamp_(
                min=torch.zeros((1, 4), device=boxes.device),
                max=torch.tensor([[clamp_shape[1], clamp_shape[0], clamp_shape[1], clamp_shape[0]]], device=boxes.device))

        # NOTE: this is in XYXY now.
        return bboxes

# Expects proposal_boxes. Outputs pred_boxes.
class DeltaBox2DPredictor(Box2DPredictor):
    def __init__(self, embed_dim, num_layers=3, size_bias=0.0):
        super(DeltaBox2DPredictor, self).__init__(embed_dim, num_layers=num_layers)

        self.transform = DeltaBox2DTransform()

    def forward(self, x, proposals, sensor):
        deltas = self.mlp(x)
        for deltas_, proposals_ in zip(deltas, proposals):
            proposals_.pred_boxes_delta = deltas_
            proposals_.pred_boxes = box_xyxy_to_cxcywh(self.transform.apply_deltas(
                deltas_,
                proposals_.proposal_boxes,
                # Might make sense to remove ViT padding and use .image_sizes here.
                clamp_shape=sensor["image"].data.tensor.shape[-2:]
            ))
            

        return x

class AbsoluteBox3DPredictor(Predictor):
    def __init__(
        self,
        embed_dim, 
        pose_type="z", # 姿态参数化方式；当前实现用 "z" 表示重力对齐、只学习一个角（yaw）
        z_type="direct", # 深度参数化方式（当前代码未使用；是个“占位”接口）
        scale_shift=True, # 是否对网络输出的 z / dims 做反白化（scale/shift）恢复到物理尺度
        num_layers=3,
        pred_proj_rel_pred_box=True,  # 2D 中心偏移是相对 pred_boxes 还是 proposal_boxes 解码
        clamp_xy_to_border=True): # 是否把 2D 投影中心 clamp 到图像边界内
        super(AbsoluteBox3DPredictor, self).__init__(embed_dim=embed_dim)

        self.pose_type = pose_type
        self.z_type = z_type
        self.scale_shift = scale_shift
        self.pred_proj_rel_pred_box = pred_proj_rel_pred_box
        self.clamp_xy_to_border = clamp_xy_to_border

        # 输出头的维度定义：
        #   dx, dy (2)  —— 2D 投影中心相对偏移（归一化到盒宽高）
        #   z       (1) —— 深度/距离（后续可能做反白化）
        #   w, l, h (3) —— 三个尺寸（后续用 exp 保证为正，并可反白化）
        #   pose    (1) —— 姿态（在重力对齐模式下只用一个角：yaw）
        self.center_2d_dim = 2
        self.z_dim = 1
        self.dims_dim = 3
        self.pose_dim = 1 #TODO: ori:1
        # 头部：MLP 从 embed_dim → embed_dim → out_dim=2+1+3+1，共 3 层（含输出层）
        self.mlp = MLP(
            embed_dim,
            embed_dim,
            self.center_2d_dim + self.z_dim + self.dims_dim + self.pose_dim,
            3)

        # 重要初始化：把最后一层中 dx,dy 的权重和偏置初始化为 0，
        # 这样一开始不会改动 2D 中心（更稳定，避免训练初期的大幅偏移）。
        # dxdy init as zero.
        nn.init.constant_(self.mlp.layers[-1].weight.data[:self.center_2d_dim], 0)
        nn.init.constant_(self.mlp.layers[-1].bias.data[:self.center_2d_dim], 0)

    def _scale_z(self, z, info):
        """
        对 z 做反白化：z_scaled = scale * z + shift
        注意：如果是 WhitenedDepthMeasurementInfo，用的是 info.parameters；
             否则用上游预测的 info.pred_parameters（更“在线”）。
        """
        if isinstance(info, WhitenedDepthMeasurementInfo):
            parameters = info.parameters
        if isinstance(info, torch.Tensor):
            parameters = info
        else:
            parameters = info.pred_parameters

        shift_parameters, scale_parameters = torch.split(parameters, 1, dim=-1)
        z = scale_parameters * z + shift_parameters
        return z
    
    

    def _scale_dims(self, z, info):
        """
        对尺寸做反白化：dims_scaled = scale * dims
        这里只做乘法（不加偏置），假设白化时只做了尺度归一化。
        """
        if isinstance(info, WhitenedDepthMeasurementInfo):
            parameters = info.parameters
        if isinstance(info, torch.Tensor):
            parameters = info
        else:
            parameters = info.pred_parameters

        _, scale_parameters = torch.split(parameters, 1, dim=-1)
        z = scale_parameters * z
        return z

    @property
    def gravity_aligned(self):
        # 当前仅当 pose_type == "z" 时表示“重力对齐”（只学习一个转角）
        return self.pose_type == "z"

    def forward(self, x, proposals, sensor):
        # x: (B, N, embed_dim)；proposals: 长度 B 的容器；sensor 提供图像/深度及其 info
        batch_size = len(x)
        # MLP 输出按顺序切分：dxdy(2) + z(1) + wlh(3) + pose(1)
        box_2d_deltas, box_z_unscaled, box_dims, box_pose = torch.split(
            self.mlp(x), (self.center_2d_dim, self.z_dim, self.dims_dim, self.pose_dim), dim=-1)
        # === 姿态矩阵构造（重力对齐，仅一个角）===
        if self.pose_type == "z":
            # Hard-code the XZ components to 0.
            # 把网络输出的单通道 box_pose 当作“yaw”，另外两个欧拉角分量强制为 0
            # 下行拼成 (yaw, 0, 0) 的形式；后续用 'YXZ' 顺序转矩阵：
            # 仅允许绕 Y 轴的旋转（注意：这依赖你的坐标系把“重力轴”定义为 Y）。
            # TODO: ori
            box_pose = torch.cat((box_pose, torch.zeros_like(box_pose), torch.zeros_like(box_pose)), dim=-1)
            box_pose = euler_angles_to_matrix(box_pose.view(-1, 3), 'YXZ').view(batch_size, -1, 3, 3) 
            
            # box_pose = euler_angles_to_matrix(box_pose.view(-1, 3), 'YXZ').view(batch_size, -1, 3, 3)  # ZYX

            
        # 选择用于反白化的 info：优先用深度通道；否则回落到图像通道
        scale_infos = sensor["depth"].info if "depth" in sensor else sensor["image"].info

        pred_scale = scale_infos[0].pred_parameters
        # === z 的反白化 ===
        if self.scale_shift:
            # 逐 batch 样本与其 info 配对做反白化；要求每个样本的 query 数 N 一致
            box_z_scaled = torch.stack([self._scale_z(box_z_unscaled_, info_) for box_z_unscaled_, info_ in zip(box_z_unscaled, pred_scale)], dim=0)
        else:
            # 不做反白化时，直接将网络输出当作已缩放值使用
            box_z_scaled = box_z_unscaled

        
        # === 尺寸正值化 +（可选）反白化 ===
        # 先裁上界再 exp，防爆；但没有下界，极小值可能导致不稳定（可按需加下界）
        box_dims = torch.exp(box_dims.clip(max=5))
        
        if self.scale_shift:
            box_dims = torch.stack([self._scale_dims(box_dims_unscaled_, info_) for box_dims_unscaled_, info_ in zip(box_dims, pred_scale)], dim=0)

        # clamp 用的图像尺寸（H, W）
        clamp_shape = sensor["image"].data.tensor.shape[-2:]
        
        info_ = scale_infos[0]
        #TODO:因为你并没有把图像真正的拼接起来
        # 逐样本写回 proposals；在这里完成 2D 投影中心解码、边界裁剪以及字段挂载
        # for box_2d_deltas_, box_z_unscaled_, box_z_scaled_, box_dims_, box_pose_, proposals_, info_ in zip(
        #         box_2d_deltas, box_z_unscaled, box_z_scaled, box_dims, box_pose, proposals, scale_infos):
        for box_2d_deltas_, box_z_unscaled_, box_z_scaled_, box_dims_, box_pose_, proposals_ in zip(
                box_2d_deltas, box_z_unscaled, box_z_scaled, box_dims, box_pose, proposals):
            # 用哪个盒来做相对偏移解码？
            if self.pred_proj_rel_pred_box:
                # 相对 pred_boxes（需要上游已写入 proposals_.pred_boxes）
                pred_proj_xy_ = proposals_.pred_boxes[:, :2] + box_2d_deltas_ * proposals_.pred_boxes[:, 2:]
            else:
                # 相对 proposal_boxes（数据集中给定的候选框）
                pred_proj_xy_ = proposals_.proposal_boxes[:, :2] + box_2d_deltas_ * proposals_.proposal_boxes[:, 2:]

            if self.clamp_xy_to_border:
                # 将 2D 投影中心 clamp 到图像边界内
                # 注意：此处上界用的是 (W, H)，若严格像素索引可考虑 (W-1, H-1)
                pred_proj_xy_.clamp_(
                    min=torch.zeros((1, 2), device=x.device),
                    max=torch.tensor([[clamp_shape[1], clamp_shape[0]]], device=x.device))
            # 将当前层的预测结果挂到 proposals_，供后续模块使用
            proposals_.pred_proj_xy = pred_proj_xy_
            proposals_.pred_z_unscaled = box_z_unscaled_
            proposals_.pred_z_scaled = box_z_scaled_
            proposals_.pred_dims = box_dims_
            proposals_.pred_pose = box_pose_

            # Hack where we move any predicted values from the info (which will be
            # overwritten by the next layer into the current layer's predictions).
            # 将 info_ 中的“预测参数/重力”迁移到 proposals_，并清空 info_ 对应字段，
            # 避免被下一层覆盖（下游模块请从 proposals_ 读取这些字段）
            if hasattr(info_, "pred_parameters"):
                proposals_._pred_parameters = info_.pred_parameters
                info_.pred_parameters = None

            if hasattr(info_, "pred_gravity"):
                proposals_._pred_gravity = info_.pred_gravity
                info_.pred_gravity = None
                
        # 返回 x（残差/后续解码仍可继续用），真正的预测已写入 proposals_
        return x

class Prompter(nn.Module):
    def __init__(self, encoders=None):
        super(Prompter, self).__init__()

        self.encoders = encoders
        self.box_2d_transform = DeltaBox2DTransform()

class Prompt(object):
    def __init__(self, query, pos_embed, self_attn_mask, pred=None, box_attn_prior_mask=None, has_output=True):
        self.query = query
        self.pos_embed = pos_embed
        self.self_attn_mask = self_attn_mask
        self.box_attn_prior_mask = box_attn_prior_mask

        if self.box_attn_prior_mask is None:
            self.box_attn_prior_mask = torch.ones((self.number_prompts,), dtype=torch.bool, device=self.device) # bool

        self.has_output = has_output

        # Indicates that _for_ this prompt, a prediction already exists and
        # should be refined. For now, we always assume it is non-None.
        self.pred = pred

    @property
    def batch_size(self):
        return self.query.shape[0]

    @property
    def number_prompts(self):
        return self.query.shape[1]

    @property
    def device(self):
        return self.query.device

    @classmethod
    def cat(cls, prompt_list):
        total_prompts = sum(prompt_.number_prompts for prompt_ in prompt_list)

        # Combine into a set of coherent prompts that can be immediately sent for decoding.
        prompt_full_self_attn = torch.ones(
            (total_prompts, total_prompts), dtype=torch.bool, device=prompt_list[0].device)
        
        prompt_start_idx = 0
        for prompt in prompt_list:
            prompt_full_self_attn[
                prompt_start_idx:(prompt_start_idx + prompt.number_prompts),
                prompt_start_idx:(prompt_start_idx + prompt.number_prompts)] = prompt.self_attn_mask

            prompt_start_idx += prompt.number_prompts

        # Purposefully returns a generic Prompt for now.
        return Prompt(
            torch.cat([prompt_.query for prompt_ in prompt_list], dim=1),
            torch.cat([prompt_.pos_embed for prompt_ in prompt_list], dim=1),
            prompt_full_self_attn,
            box_attn_prior_mask=torch.cat([prompt_.box_attn_prior_mask for prompt_ in prompt_list]),
            pred=None) # No way to assume we can concatenate instances?

class MetricQueries(Prompter):
    class MetricPrompt(Prompt):
        def __init__(self, query, pos_embed, self_attn_mask):
            super(MetricQueries.MetricPrompt, self).__init__(
                # NOTE: These attend to the global feature maps during cross attention.
                query, pos_embed, self_attn_mask, None, box_attn_prior_mask=torch.zeros((2,), dtype=torch.bool, device=query.device), # bool
                has_output=False)

    def __init__(self, input_channels, input_stride, predictors, encoders=None):
        super(MetricQueries, self).__init__()

        self.input_channels = input_channels
        self.input_stride = input_stride
        self.predictors = predictors

        self.embed_dim = self.input_channels
        # We add two tokens to predict the scale and shift of the GT depth map.
        self.query_embed = nn.Embedding(2, self.embed_dim)

    def forward(self, src_flatten, mask_flatten, spatial_shapes, sensor):
        batch_size = src_flatten.shape[0]
        metric_queries = self.query_embed.weight[None].repeat(batch_size, 1, 1)

        # Prevent attention between ordinary queries and hybrid queries.
        self_attn_mask = torch.zeros((2, 2), dtype=torch.bool, device=metric_queries.device) # bool

        return MetricQueries.MetricPrompt(
            query=metric_queries,
            pos_embed=torch.zeros_like(metric_queries),
            self_attn_mask=self_attn_mask)

    def inference(self, prompt, output, sensor, top_k=100):
        # No-op since the scale should already be applied by now.
        return None

class EncoderProposals(Prompter):
    class EncoderPrompt(Prompt):
        def __init__(self, query, pos_embed, self_attn_mask, encoder_proposals, encoder_preds, num_one2one):
            super(EncoderProposals.EncoderPrompt, self).__init__(query, pos_embed, self_attn_mask, encoder_preds)

            self.encoder_proposals = encoder_proposals
            self.num_one2one = num_one2one

    def __init__(self,
                 input_channels,
                 input_stride,
                 level_strides,
                 predictors,
                 min_size=50,
                 top_k_test=None,
                 encoders=None):
        super(EncoderProposals, self).__init__()

        self.embed_dim = input_channels
        self.input_stride = input_stride
        self.level_strides = level_strides
        self.predictors = nn.ModuleList(predictors)

        self.min_proposal_size = min_size

        self.top_k_test = top_k_test

        self.num_classes = next(p for p in self.predictors if isinstance(p, ClassPredictor)).num_classes

        self.query_embed = nn.Embedding(1200, self.embed_dim)

        # This is _somewhat_ surprising, but in mixed selection mode (our default), the
        # encoder 2D box predictions provide the positional embedding, but the content embedding
        # is still learned. It's hard to wrap my head around the fact that the assignment of content
        # to positional encoding is purely determined by topk() and not by some spatial prior usually
        # associated with DETR content queries.
        if self.number_levels > 1:
            self.enc_output_proj = nn.ModuleList([])
            for level_stride in self.level_strides:
                if level_stride == self.input_stride:
                    self.enc_output_proj.append(nn.Identity())
                elif level_stride > self.input_stride:
                    scale = int(math.log2(level_stride / self.input_stride))
                    layers = []
                    for _ in range(scale - 1):
                        layers += [
                            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
                            LayerNorm2D(self.embed_dim),
                            nn.GELU()
                        ]
                    layers.append(nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2))
                    self.enc_output_proj.append(nn.Sequential(*layers))
                else:
                    scale = int(math.log2(self.input_stride / level_stride))
                    layers = []
                    for _ in range(scale - 1):
                        layers += [
                            nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2),
                            LayerNorm2D(d_model),
                            nn.GELU()
                        ]
                    layers.append(nn.ConvTranspose2d(self.embed_dim, self.embed_dim, kernel_size=2, stride=2))
                    self.enc_output_proj.append(nn.Sequential(*layers))

        self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)

        self.prefix = "encp_"

    @property
    def number_levels(self):
        return len(self.level_strides)

    def forward(self, src_flatten, mask_flatten, spatial_shapes, sensor):
        # We consider the parameter-free _proposals_ from the encoder and then the subsequently
        # refined _instances_.
        encoder_proposals, pred_instances = self.get_proposals(src_flatten, mask_flatten, spatial_shapes, sensor)

        # Assume uniform number of boxes.
        # 这个是一个专门的位置编码网络（类名叫 Box2DPromptEncoderLearned），用于把每个框 (cx, cy, w, h) 映射成对应的positional embedding 向量
        # will clamp 2d box proposals using the clamp function as max_x=1280, max_y=1280, max_w=1280, max_h=1280
        box_2d_pos_embed = self.encoders.box_2d_encoder(torch.stack([
            pred_instances_.pred_boxes.detach() for pred_instances_ in pred_instances
        ], dim=0))

        batch_size, number_queries, _ = box_2d_pos_embed.shape
        box_2d_queries = self.query_embed.weight[None, :number_queries].repeat(batch_size, 1, 1) #生成“可学习的 query 向量”，
        self_attn_mask = torch.zeros((number_queries, number_queries), dtype=torch.bool, device=box_2d_queries.device) # bool
        #这里全是 False（0），意思是所有 queries 之间都可以互相注意，不屏蔽任何连接。
        # (query pos_embed self_attn_mask)共同构成 Decoder 的输入 token 序列。
        return EncoderProposals.EncoderPrompt(
            query=box_2d_queries, #可学习的对象原型（语义）
            pos_embed=box_2d_pos_embed, #来自 Encoder 的 2D 框位置编码（几何）
            self_attn_mask=self_attn_mask, #控制它们能否互相通信（允许全部通信）
            encoder_proposals=encoder_proposals,
            encoder_preds=pred_instances,
            num_one2one=self.top_k_test)

    def expand_encoder_output(self, memory, memory_padding_mask, spatial_shapes):
        assert spatial_shapes.size(0) == 1, f'Get encoder output of shape {spatial_shapes}, not sure how to expand'

        bs, _, c = memory.shape

        _out_memory = memory.view(bs, spatial_shapes[0, 0], spatial_shapes[0, 1], c).permute(0, 3, 1, 2)
        _out_memory_padding_mask = memory_padding_mask.view(bs, spatial_shapes[0, 0], spatial_shapes[0, 1])

        out_memory, out_memory_padding_mask, out_spatial_shapes = [], [], []
        start_level = 0
        for i in range(self.number_levels):
            if i < start_level:
                continue

            mem = self.enc_output_proj[i](_out_memory)
            mask = F.interpolate(
                _out_memory_padding_mask[None].float(), size=mem.shape[-2:]
            ).to(torch.bool)

            out_memory.append(mem)
            out_memory_padding_mask.append(mask.squeeze(0))
            out_spatial_shapes.append(mem.shape[-2:])

        out_memory = torch.cat([mem.flatten(2).transpose(1, 2) for mem in out_memory], dim=1)
        out_memory_padding_mask = torch.cat([mask.flatten(1) for mask in out_memory_padding_mask], dim=1)
        out_spatial_shapes = torch.as_tensor(out_spatial_shapes, dtype=torch.long, device=out_memory.device)

        return out_memory, out_memory_padding_mask, out_spatial_shapes

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes, sensor):
        # get multi-level features
        if self.number_levels > 1: 
            memory, memory_padding_mask, spatial_shapes = self.expand_encoder_output(
                memory, memory_padding_mask, spatial_shapes
            )

        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        start_level = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            lvl = lvl + start_level
            stride = self.level_strides[lvl]

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) * stride
            wh = torch.ones_like(grid) * self.min_proposal_size * (2.0 ** (lvl - start_level))
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)

        H_, W_ = spatial_shapes[0]
        stride = self.level_strides[0]
        mask_flatten_ = memory_padding_mask[:, :H_*W_].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1, keepdim=True) * stride
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1, keepdim=True) * stride
        img_size = torch.cat([valid_W, valid_H, valid_W, valid_H], dim=-1)
        img_size = img_size.unsqueeze(1) # [BS, 1, 4]

        # Should these lines be run during inference?
        output_proposals_valid = (
            (output_proposals > 0.01 * img_size) & (output_proposals < 0.99 * img_size)
        ).all(-1, keepdim=True)
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1).repeat(1, 1, 1),
            max(H_, W_) * stride,
        )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid,
            max(H_, W_) * stride,
        )
        # get multi-level trid 2d proposals
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory)) #0经过Norm之后就不是0了

        return output_memory, output_proposals

    def get_proposals(self, memory, mask_flatten, spatial_shapes, sensor):
        output_memory, box_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes, sensor) #sensor无影响
        # TODO
        image_size = sensor["image"].data.image_sizes[0]
        # according to the batch size B
        encoder_proposals = [Instances3D(image_size) for _ in range(memory.shape[0])]
        # encoder_proposals = [Instances3D(image_size) for image_size in sensor["image"].data.image_sizes]
        for encoder_proposals_, box_proposals_ in zip(encoder_proposals, box_proposals):
            # We call these anchors because they follow some spatial prior.
            encoder_proposals_.proposal_boxes = box_proposals_

        for predictor in self.predictors:
            predictor(output_memory, encoder_proposals, sensor)

        # NOTE: for hybrid training, we will have a much larger topk.
        top_k = self.top_k_test

        instances = []
        for encoder_proposal in encoder_proposals:
            # index 0 is foreground-ness.
            topk_proposals_ = torch.topk(encoder_proposal.pred_logits[..., 0], min(top_k, len(encoder_proposal)), dim=0)[1] # chenged to 0 26-1-6-0039
            instances_ = encoder_proposal.clone()[topk_proposals_]

            instances.append(instances_)

        # Hack: shove the proposals (needed for encoder supervision) and the topk instances
        #       (which are needed for initialization of queries).
        return (encoder_proposals, instances)

    

    def inference_single_image(self, output, image_size, topk):           
        #TODO: ori: 
        class_prob = output.pred_logits.sigmoid() 
        topk_values, topk_indexes = torch.topk(class_prob.view(-1), topk)
        labels = topk_indexes % class_prob.shape[-1]
        ################################################
        class_prob = output.pred_logits.sigmoid()[:, 1]
        topk_values, topk_indexes = torch.topk(class_prob.view(-1), topk)

        class_scores = topk_values
        topk_boxes = topk_indexes 
        # labels = topk_indexes % class_prob.shape[-1]
        
        boxes = box_cxcywh_to_xyxy(output.pred_boxes)
        boxes = boxes[topk_boxes]
        xyz = output.pred_xyz[topk_boxes]
        dims = output.pred_dims[topk_boxes]
        pose = output.pred_pose[topk_boxes]
        object_desc = output.object_desc[topk_boxes]
        proj_xy = output.pred_proj_xy[topk_boxes]

        result = Instances3D(image_size)
            
        result.scores = class_scores
        result.pred_classes = labels
        result.pred_boxes = boxes
        result.pred_logits = output.pred_logits[topk_boxes]
        result.pred_boxes.clip_(
            min=torch.tensor([0.0, 0.0, 0.0, 0.0], device=boxes.device),
            max=torch.tensor([image_size[1], image_size[0], image_size[1], image_size[0]], device=boxes.device))

        result.pred_boxes_3d = GeneralInstance3DBoxes(
            # Account for WHL ordering.
            torch.cat((xyz, dims[:, [2, 1, 0]]), dim=-1),
            pose)
        result.object_desc = object_desc
        result.pred_proj_xy = proj_xy
        
        return result
    
    def train_single_image(self, output, image_size, topk):
        """
        不排序、不选 topk：保留所有 query，保持原始顺序。
        仍然保持同样的函数签名/返回类型，topk 参数会被忽略（为了兼容接口）。
        """
        # 前景分数（假设二分类，前景在 logit[:,1]）
        # 如果你的 pred_logits 不是 [Nq,2]，你需要改成取你想用的那个类别
        class_prob = output.pred_logits.sigmoid()[:, 1]        # [Nq]
        class_scores = class_prob                               # 不排序

        # labels：保持接口字段存在
        # 二分类时你如果只输出“前景候选”，可以全设为 1
        labels = torch.ones_like(class_scores, dtype=torch.long)

        # 不做 topk 过滤：全部保留
        boxes = box_cxcywh_to_xyxy(output.pred_boxes)          # [Nq,4]
        xyz = output.pred_xyz                                  # [Nq,3] or [Nq,*]
        dims = output.pred_dims                                # [Nq,3]
        pose = output.pred_pose                                # [Nq,*]
        object_desc = output.object_desc                        # [Nq,C]
        proj_xy = output.pred_proj_xy                          # [Nq,2] or [Nq,*]

        result = Instances3D(image_size)

        result.scores = class_scores
        result.pred_classes = labels
        result.pred_boxes = boxes
        result.pred_logits = output.pred_logits

        result.pred_boxes.clip_(
            min=torch.tensor([0.0, 0.0, 0.0, 0.0], device=boxes.device),
            max=torch.tensor([image_size[1], image_size[0], image_size[1], image_size[0]], device=boxes.device),
        )

        result.pred_boxes_3d = GeneralInstance3DBoxes(
            torch.cat((xyz, dims[:, [2, 1, 0]]), dim=-1),
            pose
        )
        result.object_desc = object_desc
        result.pred_proj_xy = proj_xy

        return result
    
    def inference(self, output, sensor, topk, intrinsic, extrinsic, gravity):
        results = []
        N_batch = intrinsic.shape[0]
        N_img = intrinsic.shape[1]
        intrinsic = intrinsic.reshape(-1,3,3)
        extrinsic = extrinsic.reshape(-1,3,4)
        # 创建底行 [0, 0, 0, 1]
        bottom = torch.tensor([0, 0, 0, 1], dtype=extrinsic.dtype, device=extrinsic.device)
        bottom = bottom.view(1, 1, 4).repeat(extrinsic.shape[0], 1, 1)  # [N,1,4]
        # 拼接到 extrinsic 下方
        extrinsic_cat = torch.cat([extrinsic, bottom], dim=1)  # [N,4,4]
        extrinsic_homo = torch.linalg.inv(extrinsic_cat)
        
        # output的第一维度应该是N_batch * N_img
        batch_output = []
        for index, output_ in enumerate(output):
            # 遍历每一张图像
            # info = sensor["image"].sensor[index]

            # K = info.image.K[-1:].repeat(len(output_), 1, 1)
            K = intrinsic[index:index+1].repeat(len(output_), 1, 1)
            
            output_xyz = torch.bmm(
                torch.linalg.inv(K), torch.cat((output_.pred_z_scaled * output_.pred_proj_xy, output_.pred_z_scaled), dim=-1)[..., None])[..., 0]
            
                
            # OPTION 3:
            # using the T_gravity from gravity head
            output_.pred_pose = gravity[index//N_img, index % N_img].unsqueeze(0) @ output_.pred_pose
            
            
            # Transform to world space
            R = extrinsic_homo[index, :3, :3]
            t = extrinsic_homo[index, :3, 3]
            output_xyz_world = (output_xyz @ R.T) + t
            # output_xyz_world = (R.unsqueeze(0) @ output_xyz ) + t
            output_.pred_pose = torch.matmul(R[None, :, :], output_.pred_pose)
            output_.pred_xyz = output_xyz_world  #output_xyz
            # output_.pred_xyz = output_xyz  # debug
            
            
            batch_output.append(output_)
                
            if (index+1) % N_img==0:
                # batch_output = batch_output[1]
                batch_output = Instances3D.cat(batch_output)
                # if self.training:
                #     results.append(self.train_single_image(batch_output, sensor["image"].data.image_sizes[index], topk=topk))
                # else:
                # topk = min(topk, len(batch_output))
                topk = len(batch_output)
                results.append(self.inference_single_image(batch_output, sensor["image"].data.image_sizes[index], topk=topk))
                batch_output=[]

        return results

class PromptEncoders(nn.Module):
    def __init__(self, **kwargs):
        super(PromptEncoders, self).__init__()

        for encoder_name, encoder in kwargs.items():
            setattr(self, encoder_name, encoder)

class CubifyAnythingPrompting(nn.Module):
    def __init__(self, embed_dim, prompters, encoders):
        super(CubifyAnythingPrompting, self).__init__()

        self.embed_dim = embed_dim
        self.prompters = nn.ModuleList(prompters)
        self.encoders = encoders

        for prompter in self.prompters:
            prompter.encoders = self.encoders

    def get_image_prompts(
            self,
            src_flatten,
            mask_flatten,
            spatial_shapes,
            sensor,
            N_batch,
            N_img,
            embedding_dim):

        prompts = []
        prompts_0 = []
        prompts_1 = []
        src_flatten = src_flatten.reshape(N_batch*N_img, -1, embedding_dim)
        mask_flatten = mask_flatten.reshape(N_batch*N_img, -1)
        for prompter in self.prompters:
    
            prompt = prompter(src_flatten, mask_flatten, spatial_shapes, sensor)
            if prompt is not None:
                prompts.append(prompt)

        return prompts

    def get_instance_prompts(
            self,
            instances):
        prompts = []
        for prompter in self.prompters:
            if not isinstance(prompter, InstancePrompter):
                continue
            
            prompt = prompter(instances)
            if prompt is not None:
                prompts.append(prompt)
                
        return prompts
        
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

class Joiner(nn.Sequential):
    def __init__(self, backbone):
        super().__init__(backbone)

    @property
    def backbone(self):
        return self[0]

    @property
    def strides(self):
        return self.backbone.strides

    @property
    def num_channels(self):
        return self.backbone.num_channels

    @property
    def size_divisibility(self):
        return self.backbone.size_divisibility

    @property
    def _square_pad(self):
        return self.backbone._square_pad

    @property
    def padding_constraints(self):
        return self.backbone.padding_constraints

    def forward(self, sensor):
        xs = self[0](sensor)

        out = []
        for name, x in sorted(xs.items()):
            # out.append(
            #     NestedTensor(x, mask=torch.zeros((x.shape[0], x.shape[-2], x.shape[-1]), dtype=torch.bool, device=x.device)))
            # TODO: change the mask
            p_mask = torch.ones(x.shape[0], x.shape[-2], x.shape[-1], dtype=torch.bool, device=x.device)
            p_mask[:, :sensor["image"].data.image_sizes[0][0]//16, :sensor["image"].data.image_sizes[0][1]//16] = False
            out.append(
                NestedTensor(x, mask=p_mask))

        return out
    

class FusionProj(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, dropout=0.1):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or (in_dim // 2)
        # 一个轻量 MLP (可替换为 1x1 conv、或 attention)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        # 可选：残差比例
        self.gamma = nn.Parameter(torch.tensor(0.0))  # start as 0 -> identity behavior

    def forward(self, a, b):
        # a, b: [B, N, D] each, cat -> [B, N, 2D]
        x = torch.cat([a, b], dim=-1)
        out = self.proj(x)  # [B, N, D]
        # 残差混合：开始时 y 被抑制（gamma=0），训练中逐步学会利用 y
        # out = (1 - torch.sigmoid(self.gamma)) * b + torch.sigmoid(self.gamma) * y
        return out


'''
added by lyq
'''
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class VGGTMerger(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.input_dim = context_dim * (spatial_merge_size**2)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
        return x




def make_vggt():
    #load vggt 
    # vggt = VGGT.from_pretrained(vggt_path)  # 3D视觉特征提取器
    vggt = VGGT.from_pretrained("facebook/VGGT-1B")
    vggt.eval()
    vggt = vggt.to("cuda")
    
    vggt.camera_head = None  # 禁用不需要的头部
    vggt.track_head = None
    vggt.depth_head = None
    vggt.point_head = None
    for param in vggt.parameters():
        param.requires_grad = False  # 冻结VGGT参数


    return vggt

class LightweightCrossViewFusion(nn.Module):
    """
    轻量级跨视角融合模块
    计算量更小，适合资源受限的情况
    """
    def __init__(self, feat_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        
        # 简化的view embedding
        self.view_proj = nn.Linear(feat_dim, feat_dim)
        self.pos_proj = nn.Linear(feat_dim, feat_dim)
        
        # 单层注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 权重门控
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(),
            nn.Linear(feat_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(feat_dim)
    
    def forward(self, features):
        """
        Args:
            features: [N, 1024, 256]
        Returns:
            fused: [1, 1024, 256]
        """
        N, L, C = features.shape
        
        # 展平并添加view信息
        view_ids = torch.arange(N, device=features.device).float().unsqueeze(-1) / N
        view_info = self.view_proj(view_ids.unsqueeze(-1).expand(-1, -1, C))  # [N, 1, C]
        view_features = features + view_info.expand(-1, L, -1)  # [N, L, C]
        
        # 重新排列为 [L, N, C] 便于位置级注意力
        tokens = view_features.permute(1, 0, 2)  # [L, N, C]
        
        # 每个位置独立进行跨视角注意力
        fused_tokens = []
        for pos in range(L):
            pos_tokens = tokens[pos:pos+1]  # [1, N, C]
            attended, _ = self.cross_attention(pos_tokens, pos_tokens, pos_tokens)
            
            # 加权聚合
            weights = self.gate(attended)  # [1, N, 1]
            weights = F.softmax(weights.squeeze(-1), dim=-1)  # [1, N]
            aggregated = (attended * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)  # [1, 1, C]
            fused_tokens.append(aggregated)
        
        fused = torch.cat(fused_tokens, dim=1)  # [1, L, C]
        fused = self.norm(fused)
        
        return fused

class CrossViewSelfAttentionFusion(nn.Module):
    """
    Cross-View Self-Attention特征融合模块
    将多视角特征展平为token序列，附加view embedding，通过transformer融合
    
    输入: [N, 1024, 256] - N个视角的特征
    输出: [1, 1024, 256] - 融合后的全景特征
    """
    def __init__(self, 
                 feat_dim=256, 
                 num_heads=8, 
                 num_layers=3,
                 dropout=0.1,
                 use_pos_embed=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_pos_embed = use_pos_embed
        
        # View Embedding - 为每个视角学习可区分的embedding
        self.max_views = 16  # 支持最大视角数
        self.view_embedding = nn.Embedding(self.max_views, feat_dim)
        
        # Position Embedding - 为每个token位置学习位置编码
        if self.use_pos_embed:
            self.position_embedding = nn.Embedding(1024, feat_dim)  # 假设最大1024个token
        
        # Multi-layer Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm结构，更稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(feat_dim)
        )
        
        # View-wise attention pooling
        self.view_attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 可学习的全局query用于跨视角聚合
        self.global_query = nn.Parameter(torch.randn(1, 1024, feat_dim))
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim)
        )
        
        # 残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # Xavier初始化
        nn.init.xavier_uniform_(self.global_query)
        nn.init.xavier_uniform_(self.view_embedding.weight)
        if self.use_pos_embed:
            nn.init.xavier_uniform_(self.position_embedding.weight)
        
        # 输出投影层初始化
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def add_embeddings(self, features):
        """
        为特征添加view embedding和position embedding
        Args:
            features: [N, L, C] - N个视角，每个视角L个token
        Returns:
            embedded_features: [N*L, C] - 添加embedding后的特征
            attention_mask: [N*L] - 注意力掩码
        """
        N, L, C = features.shape
        device = features.device
        
        # 1. View Embedding - 每个视角的所有token共享同一个view embedding
        view_ids = torch.arange(N, device=device)  # [N]
        view_embeds = self.view_embedding(view_ids)  # [N, C]
        view_embeds = view_embeds.unsqueeze(1).expand(-1, L, -1)  # [N, L, C]
        
        # 2. Position Embedding - 每个token位置的embedding
        if self.use_pos_embed:
            pos_ids = torch.arange(L, device=device)  # [L]
            pos_embeds = self.position_embedding(pos_ids)  # [L, C]
            pos_embeds = pos_embeds.unsqueeze(0).expand(N, -1, -1)  # [N, L, C]
        else:
            pos_embeds = 0
        
        # 3. 组合所有embedding
        embedded_features = features + view_embeds + pos_embeds  # [N, L, C]
        
        # 4. 展平为token序列
        embedded_features = embedded_features.view(N * L, C)  # [N*L, C]
        
        # 5. 创建attention mask (所有token都参与注意力)
        attention_mask = torch.zeros(N * L, dtype=torch.bool, device=device)
        
        return embedded_features, attention_mask
    
    def cross_view_attention(self, embedded_tokens):
        """
        跨视角自注意力计算
        Args:
            embedded_tokens: [1, N*L, C] - 展平的token序列
        Returns:
            attended_tokens: [1, N*L, C] - 注意力后的特征
        """
        # 通过transformer进行跨视角和跨位置的注意力计算
        attended_tokens = self.transformer(embedded_tokens)  # [1, N*L, C]
        return attended_tokens
    
    def aggregate_views(self, attended_tokens, N, L):
        """
        将跨视角注意力后的token聚合为单一视角
        Args:
            attended_tokens: [1, N*L, C] - 注意力后的token
            N: 视角数
            L: 每个视角的token数
        Returns:
            aggregated: [1, L, C] - 聚合后的单视角特征
        """
        # 重新组织为 [N, L, C] 形状
        tokens_reshaped = attended_tokens.view(1, N, L, -1)  # [1, N, L, C]
        tokens_reshaped = tokens_reshaped.squeeze(0)  # [N, L, C]
        
        # 使用可学习的全局query进行view-wise聚合
        global_query = self.global_query.expand(1, -1, -1)  # [1, L, C]
        
        # 将所有视角的对应位置token作为key/value
        # 重新排列：[N, L, C] -> [L, N, C] -> [L*N, C] -> [1, L*N, C]
        kv_tokens = tokens_reshaped.permute(1, 0, 2).contiguous()  # [L, N, C]
        kv_tokens = kv_tokens.view(L * N, -1).unsqueeze(0)  # [1, L*N, C]
        
        # 对每个位置聚合不同视角的信息
        aggregated_list = []
        for pos in range(L):
            # 提取位置pos处所有视角的token
            pos_tokens = tokens_reshaped[:, pos:pos+1, :].transpose(0, 1)  # [1, N, C]
            pos_query = global_query[:, pos:pos+1, :]  # [1, 1, C]
            
            # 注意力聚合
            aggregated_pos, _ = self.view_attention(
                query=pos_query,      # [1, 1, C]
                key=pos_tokens,       # [1, N, C]  
                value=pos_tokens      # [1, N, C]
            )
            aggregated_list.append(aggregated_pos)
        
        # 拼接所有位置的聚合结果
        aggregated = torch.cat(aggregated_list, dim=1)  # [1, L, C]
        
        return aggregated
    
    def forward(self, features):
        """
        前向传播
        Args:
            features: [N, 1024, 256] - 多视角特征
        Returns:
            fused_features: [1, 1024, 256] - 融合后的特征
        """
        N, L, C = features.shape
        device = features.device
        
        # 1. 添加view和position embedding
        embedded_tokens, attention_mask = self.add_embeddings(features)  # [N*L, C]
        embedded_tokens = embedded_tokens.unsqueeze(0)  # [1, N*L, C]
        
        # 2. 跨视角自注意力
        attended_tokens = self.cross_view_attention(embedded_tokens)  # [1, N*L, C]
        
        # 3. 聚合多视角为单视角
        aggregated_features = self.aggregate_views(attended_tokens, N, L)  # [1, L, C]
        
        # 4. 输出投影
        fused_features = self.output_proj(aggregated_features)  # [1, L, C]
        
        # 5. 残差连接到平均特征（提升稳定性）
        mean_features = features.mean(dim=0, keepdim=True)  # [1, L, C]
        final_features = fused_features + self.residual_weight * mean_features
        
        return final_features


class AttentionFusionWithTorch(nn.Module):
    def __init__(self, embed_dim=256, num_heads=1, dropout=0.1):
        """
        使用PyTorch内置MultiheadAttention的特征融合模块
        Args:
            embed_dim: 特征维度 (默认256)
            num_heads: 注意力头数 (默认1)
            dropout: Dropout概率 (默认0.1)
        """
        super().__init__()
        # 初始化多头注意力层 [6,7](@ref)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 输入输出为(batch, seq, features)格式
        )
        # 可学习的查询向量 [2](@ref)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, features):
        """
        前向传播
        Args:
            features: 输入特征 [N, 1024, 256]
        Returns:
            融合特征 [1, 1024, 256]
        """
        # 调整输入维度: [N, 1024, 256] -> [1024, N, 256]
        features_perm = features.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        
        # 扩展查询向量 (需匹配序列长度)
        query = self.query.expand(features_perm.size(0), -1, -1)  # [1024, 1, 256]
        
        # 注意: key/value 使用调整后的 features_perm
        fused, _ = self.multihead_attn(
            query=query,          # [1024, 1, 256]
            key=features_perm,    # [1024, N, 256]
            value=features_perm, # [1024, N, 256]
            need_weights=False
        )
        # 输出形状: [1024, 1, 256] -> [1, 1024, 256]
        return fused.permute(1, 0, 2)



class CrossModalMultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dropout=0.1):
        super().__init__()
        # 投影层：将特征B的维度从2048降到256（与A对齐）
        self.proj_b = nn.Linear(2048, d_model)
        
        # PyTorch官方多头注意力模块（设置batch_first适配[N, seq, feat]格式）
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, # 输入/输出维度
            num_heads=num_heads, # 头数（256÷8=32，符合整除要求）
            dropout=dropout, # 输入格式为 [N, seq, feat]
            batch_first=True  # 关键：匹配输入维度顺序
        )
        
        # 层归一化（稳定训练）
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, A, B):
        """
        Args:
            A: [N, 1024, 256] 作为目标特征（Query）
            B: [N, 306, 2048] 作为源特征（Key/Value）
        Returns:
            aligned_B: [N, 1024, 256] 与A对齐的特征
        """
        # 1. 投影特征B至A的维度空间 [N, 306, 2048] → [N, 306, 256]
        B_proj = self.proj_b(B)
        
        # 2. 跨模态注意力计算（A作为Query，B_proj作为Key/Value）
        attn_output, _ = self.multihead_attn(
            query=A, # [N, 1024, 256]
            key=B_proj, # [N, 306, 256]
            value=B_proj
        )
        
        # 3. 残差连接 + 层归一化（保留原始A的信息）
        return self.layer_norm(attn_output + A) # [N, 1024, 256]


class FeatureFusionModule_v1(nn.Module):
    def __init__(self, 
                 in_channels=256, 
                 out_channels=2048,
                 fusion_type='add',
                 fusion_weights=None):
        """
        重构特征融合模块：将image_embeds_3d向features_A对齐
        Args:
            in_channels: features_A的通道数 (默认256)
            out_channels: image_embeds_3d的原始通道数 (默认2048)
            fusion_type: 融合方式 'add' 或 'weighted'
            fusion_weights: 加权融合的权重 (默认[0.7, 0.3])
        """
        super().__init__()
        self.fusion_type = fusion_type
        
        # 通道对齐层：将image_embeds_3d的通道数降至features_A的维度
        self.channel_aligner = nn.Sequential(
            nn.Linear(out_channels, in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 初始化权重
        self._init_weights()
        
        # 设置融合权重
        if fusion_weights is None:
            self.fusion_weights = [0.7, 0.3] if fusion_type == 'weighted' else None


    def _init_weights(self):
        """ 初始化线性层权重 """
        nn.init.kaiming_normal_(self.channel_aligner[0].weight, 
                                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.channel_aligner[0].bias, 0)

    def forward(self, features_A, image_embeds_3d):
        # 步骤1: 通道对齐 [1*N, F_vggt, 2048] -> [1*N, F_vggt, 256]
        image_aligned = self.channel_aligner(image_embeds_3d.to(torch.float32))
        
        # 步骤2: 维度调整 [1*N, F_vggt, 256] -> [1*N, 256, F_vggt]
        image_aligned = image_aligned.permute(0, 2, 1)
        
        # 步骤3: 添加虚假的宽度维度 [1*N, 256, F_vggt] -> [1*N, 256, F_vggt, 1]
        image_aligned = image_aligned.unsqueeze(-1)
        
        # 步骤4: 序列长度插值 [F_vggt -> 1024]
        image_resized = F.interpolate(
            image_aligned, 
            size=(features_A.shape[1], 1),  # 目标尺寸: 1024x1
            mode='bilinear',
            align_corners=False
        )
        
        # 步骤5: 恢复原始维度 [1*N, 256, 1024, 1] -> [1*N, 1024, 256]
        image_resized = image_resized.squeeze(-1)      # 移除虚假宽度维度
        image_resized = image_resized.permute(0, 2, 1) # [1*N, 1024, 256]
        
        # 步骤6: 特征融合（与features_A相同形状）
        if self.fusion_type == 'add':
            fused = features_A + image_resized
        elif self.fusion_type == 'weighted' and self.fusion_weights is not None:
            fused = (self.fusion_weights[0] * features_A + 
                     self.fusion_weights[1] * image_resized)
        else:  # 拼接融合
            fused = torch.cat((features_A, image_resized), dim=-1)
        
        return fused

    def extra_repr(self):
        """ 显示模块配置信息 """
        return (f"fusion_type={self.fusion_type}, "
                f"fusion_weights={self.fusion_weights.tolist() if self.fusion_weights is not None else None}")
        
    # # 模拟输入数据
    # features_A = torch.randn(1024, 256)   # 待融合特征
    # image_embeds_3d = torch.randn(1596, 2048)  # 目标特征

    # # 特征融合
    # fused_features = fusion_module(features_A, image_embeds_3d)
    # print(f"输出尺寸: {fused_features.shape}")  # torch.Size([1596, 2048])

class FeatureFusionModule_v2(nn.Module):
    def __init__(self, 
                 in_channels=2048, 
                 out_channels=256,
                 num_heads=8,
                 dropout=0.1,
                 fusion_type='add',
                 fusion_weights=None):
        """
        重构特征融合模块：将image_embeds_3d向features_A对齐
        Args:
            in_channels: features_A的通道数 (默认256)
            out_channels: image_embeds_3d的原始通道数 (默认2048)
            fusion_type: 融合方式 'add' 或 'weighted'
            fusion_weights: 加权融合的权重 (默认[0.7, 0.3])
        """
        super().__init__()
        self.proj_b = nn.Linear(in_channels, out_channels)
        
        # PyTorch官方多头注意力模块（设置batch_first适配[N, seq, feat]格式）
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=out_channels, # 输入/输出维度
            num_heads=num_heads, # 头数（256÷8=32，符合整除要求）
            dropout=dropout, # 输入格式为 [N, seq, feat]
            batch_first=True  # 关键：匹配输入维度顺序
        )
        
        # 层归一化（稳定训练）
        self.layer_norm = nn.LayerNorm(out_channels)

        self.fusion_type = fusion_type
        # 设置融合权重
        if fusion_weights is None:
            self.fusion_weights = [0.7, 0.3] if fusion_type == 'weighted' else None

    def forward(self, A, B):
        """
        Args:
            A: [N, 1024, 256] 作为目标特征（Query）
            B: [N, 306, 2048] 作为源特征（Key/Value）
        Returns:
            aligned_B: [N, 1024, 256] 与A对齐的特征
        """
        # 1. 投影特征B至A的维度空间 [N, 306, 2048] → [N, 306, 256]
        B_proj = self.proj_b(B)
        
        # 2. 跨模态注意力计算（A作为Query，B_proj作为Key/Value）
        attn_output, _ = self.multihead_attn(
            query=A, # [N, 1024, 256]
            key=B_proj, # [N, 306, 256]
            value=B_proj # [N, 306, 256]
        )
        
        # 3. 残差连接 + 层归一化（保留原始A的信息）
        alignd_features = self.layer_norm(attn_output) # [N, 1024, 256]
        
        # 步骤6: 特征融合（与features_A相同形状）
        if self.fusion_type == 'add':
            fused = A + alignd_features
        elif self.fusion_type == 'weighted' and self.fusion_weights is not None:
            fused = (self.fusion_weights[0] * A + 
                     self.fusion_weights[1] * alignd_features)
        else:  # 拼接融合
            fused = torch.cat((A, alignd_features), dim=-1)
        
        return fused

    def extra_repr(self):
        """ 显示模块配置信息 """
        return (f"fusion_type={self.fusion_type}, "
                f"fusion_weights={self.fusion_weights.tolist() if self.fusion_weights is not None else None}")
        
    # # 模拟输入数据
    # features_A = torch.randn(1024, 256)   # 待融合特征
    # image_embeds_3d = torch.randn(1596, 2048)  # 目标特征

    # # 特征融合
    # fused_features = fusion_module(features_A, image_embeds_3d)
    # print(f"输出尺寸: {fused_features.shape}")  # torch.Size([1596, 2048])
  
class FeatureFusionModule_v3(nn.Module):
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads):
        super(FeatureFusionModule_v3, self).__init__()
        
        # pre-norm
        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_encoder_norm = nn.LayerNorm(d_spatial_encoder)
        
        # projection
        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.spatial_encoder_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.spatial_encoder_value_proj = nn.Linear(d_spatial_encoder, d_attn)
        
        # cross attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)
        
        # post-norm
        self.out_norm = nn.LayerNorm(d_attn)
        
        # projection
        self.out_proj = nn.Linear(d_attn, d_clip)
        
        # dropout
        self.dropout = nn.Dropout(0.1)
        
        # features mlp
        self.fusion_proj = FusionProj(in_dim=2*d_clip, hidden_dim=d_clip, out_dim=d_clip)

    def forward(self, clip_features, spatial_encoder_features, mask_flatten):
        """
        Args:
            clip_features: [B, N, D_clip]
            spatial_encoder_features: [B, N, D_spatial_encoder]
        Returns:
            fused_features: [B, N, D_clip]
        """
        # 1. Pre-Norm 首先对两种输入特征分别进行层归一化。
        clip_features_norm = self.clip_norm(clip_features)  # [B, N, D_clip]
        spatial_encoder_features_norm = self.spatial_encoder_norm(spatial_encoder_features)  # [B, N, D_spatial_encoder]
        
        # 2. Projection 将归一化后的特征投影到相同的注意力维度d_attn。
        clip_query_proj = self.clip_query_proj(clip_features_norm)  # [B, N, D_attn]
        spatial_encoder_key_proj = self.spatial_encoder_key_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        spatial_encoder_value_proj = self.spatial_encoder_value_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        
        # 3. Cross-Attention 核心计算：使用CLIP的Query去查询空间编码器的Key和Value。
        #TODO:mask for attention fusion
        # attn_mask = torch.zeros((mask_flatten.shape[1],spatial_encoder_key_proj.shape[1]), device=clip_query_proj.device, dtype=clip_query_proj.dtype)
        # attn_mask[mask_flatten[1],:] = float('-inf')
        
        fused_features, attn_weights = self.cross_attention(
            query=clip_query_proj,
            key=spatial_encoder_key_proj,
            value=spatial_encoder_value_proj,
            # attn_mask=attn_mask #TODO:newly added
        )
        # fused_features形状: [B, N, D_attn]
        # attn_weights形状: [B, num_heads, N_query (来自CLIP), N_key (来自空间编码器)]，表示注意力分配权重
        
        # 4. Output Projection 将融合后的特征从d_attn维度投影回CLIP特征的原始维度d_clip。
        fused_features = self.out_proj(fused_features)   # [B, N_clip, D_clip]
        
        # 5. Residual Connection and Dropout 残差连接与Dropout。
        fused_features = self.out_norm(fused_features) # 对输出进行层归一化
        
        # fused_features = fused_features + clip_features  # 残差连接：将融合后的特征与原始的CLIP特征相加。这有助于梯度流动和保留原始信息。 
        fused_features = self.fusion_proj(fused_features, clip_features)
        
        # print(f'status_of_fused_features: max:{fused_features.max():.2f}, min:{fused_features.min():.2f}, mean:{fused_features.mean():.2f}, std:{fused_features.std():.2f}')
        # print(f'status_of_clip_features: max:{clip_features.max():.2f}, min:{clip_features.min():.2f}, mean:{clip_features.mean():.2f}, std:{clip_features.std():.2f}')
        fused_features = self.dropout(fused_features) # 最后应用Dropout
        
        return fused_features #, attn_weights # 返回融合后的特征和注意力权重
  
    
    
        
    
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

