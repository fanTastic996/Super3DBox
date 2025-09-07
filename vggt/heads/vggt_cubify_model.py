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
        self, src, src_pos_embed, src_padding_mask, src_spatial_shapes, level_start_index, valid_ratios,
        prompts, sensor):
        # Not the best assumption for now, but assume we can treat the "prompt" in a uniform manner as "reference.
        # Interleave so we have List[List[Instances]] of size # of instances x # of prompts
        # Original implementation:
        reference = [Instances3D.cat([prompt.pred[idx] for prompt in prompts if prompt.box_attn_prior_mask.any()])
                     for idx in range(prompts[0].batch_size)] #batch size: N
        
        # REVISED BY LYQ
        # reference = []
        # for idx in range(prompts[0].batch_size):
        #     # 创建内层列表用于收集符合条件的prompt.pred值
        #     inner_list = []
        #     for prompt in prompts: #[MetricPrompt,EncoderPrompt]
        #         # 条件过滤：只处理box_attn_prior_mask不为空的prompt
        #         #[Instances3D]
        #         if prompt.box_attn_prior_mask.any():
        #             print("prompt.pred",prompt.pred,len(prompt.pred))
        #             inner_list.append(prompt.pred[0])
            
        #     # 使用Instances3D.cat处理内层列表并添加到结果中
        #     reference.append(Instances3D.cat(inner_list))
        

        prompts = Prompt.cat(prompts)        
        output = prompts.query
        intermediate = []
        intermediate_preds = []

        for lid, layer in enumerate(self.layers):
            # Currently, the only thing that influences the RPE is the current 2D box prediction.
            reference_2d = torch.stack([ref_.pred_boxes for ref_ in reference]).detach()
            output = layer(
                output,
                prompts.pos_embed,
                # For now, we only bias the attention based on the current 2D box.
                reference_2d[:, :, None],
                src,
                src_pos_embed,
                src_spatial_shapes,
                src_padding_mask,
                prompts.self_attn_mask,
                box_attn_prior_mask=prompts.box_attn_prior_mask
            )

            output_after_norm = self.norm(output)

            pred_instances = [Instances3D(image_size) for image_size in sensor["image"].data.image_sizes]
            for pred_instances_, reference_ in zip(pred_instances, reference):
                # The previous layer's "predictions", are this rounds, "proposals"
                pred_instances_.proposal_boxes = reference_.pred_boxes
                if hasattr(pred_instances_, "pred_proj_xy"):
                    pred_instances_.proposal_proj_xy = reference_.pred_proj_xy
                    pred_instances_.proposal_z_unscaled = reference_.pred_z_unscaled
                    pred_instances_.proposal_dims = reference_.pred_dims
                    pred_instances_.proposal_pose = reference_.pred_pose

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

    def forward(self, x, proposals, sensor):
        pred_shift = torch.exp(self.shift(x[:, 0:1]))
        pred_scale = torch.exp(self.scale(x[:, 1:2]))

        shift_scale = torch.cat((pred_shift, pred_scale), dim=-1)
        # Probably better to store on "depth", but we don't have that sensor for
        # monocular.
        scale_infos = sensor["depth"].info if "depth" in sensor else sensor["image"].info
        for i, info_ in enumerate(scale_infos):
            info_.pred_parameters = shift_scale[i]

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
        pose_type="z",
        z_type="direct",
        scale_shift=True,
        num_layers=3,
        pred_proj_rel_pred_box=True,
        clamp_xy_to_border=True):
        super(AbsoluteBox3DPredictor, self).__init__(embed_dim=embed_dim)

        self.pose_type = pose_type
        self.z_type = z_type
        self.scale_shift = scale_shift
        self.pred_proj_rel_pred_box = pred_proj_rel_pred_box
        self.clamp_xy_to_border = clamp_xy_to_border

        # dxdy + z + wlh + pose
        self.center_2d_dim = 2
        self.z_dim = 1
        self.dims_dim = 3
        self.pose_dim = 1

        self.mlp = MLP(
            embed_dim,
            embed_dim,
            self.center_2d_dim + self.z_dim + self.dims_dim + self.pose_dim,
            3)

        # dxdy init as zero.
        nn.init.constant_(self.mlp.layers[-1].weight.data[:self.center_2d_dim], 0)
        nn.init.constant_(self.mlp.layers[-1].bias.data[:self.center_2d_dim], 0)

    def _scale_z(self, z, info):
        if isinstance(info, WhitenedDepthMeasurementInfo):
            parameters = info.parameters
        else:
            parameters = info.pred_parameters

        shift_parameters, scale_parameters = torch.split(parameters, 1, dim=-1)
        z = scale_parameters * z + shift_parameters
        return z

    def _scale_dims(self, z, info):
        if isinstance(info, WhitenedDepthMeasurementInfo):
            parameters = info.parameters
        else:
            parameters = info.pred_parameters

        _, scale_parameters = torch.split(parameters, 1, dim=-1)
        z = scale_parameters * z
        return z

    @property
    def gravity_aligned(self):
        return self.pose_type == "z"

    def forward(self, x, proposals, sensor):
        batch_size = len(x)
        box_2d_deltas, box_z_unscaled, box_dims, box_pose = torch.split(
            self.mlp(x), (self.center_2d_dim, self.z_dim, self.dims_dim, self.pose_dim), dim=-1)

        if self.pose_type == "z":
            # Hard-code the XZ components to 0.
            box_pose = torch.cat((box_pose, torch.zeros_like(box_pose), torch.zeros_like(box_pose)), dim=-1)
            box_pose = euler_angles_to_matrix(box_pose.view(-1, 3), 'YXZ').view(batch_size, -1, 3, 3)

        scale_infos = sensor["depth"].info if "depth" in sensor else sensor["image"].info

        if self.scale_shift:
            box_z_scaled = torch.stack([self._scale_z(box_z_unscaled_, info_) for box_z_unscaled_, info_ in zip(box_z_unscaled, scale_infos)], dim=0)
        else:
            # Assume the verbatim output is also scaled.
            box_z_scaled = box_z_unscaled

        box_dims = torch.exp(box_dims.clip(max=5))
        if self.scale_shift:
            box_dims = torch.stack([self._scale_dims(box_dims_unscaled_, info_) for box_dims_unscaled_, info_ in zip(box_dims, scale_infos)], dim=0)

        clamp_shape = sensor["image"].data.tensor.shape[-2:]
        for box_2d_deltas_, box_z_unscaled_, box_z_scaled_, box_dims_, box_pose_, proposals_, info_ in zip(
                box_2d_deltas, box_z_unscaled, box_z_scaled, box_dims, box_pose, proposals, scale_infos):
            if self.pred_proj_rel_pred_box:
                pred_proj_xy_ = proposals_.pred_boxes[:, :2] + box_2d_deltas_ * proposals_.pred_boxes[:, 2:]
            else:
                pred_proj_xy_ = proposals_.proposal_boxes[:, :2] + box_2d_deltas_ * proposals_.proposal_boxes[:, 2:]

            if self.clamp_xy_to_border:
                pred_proj_xy_.clamp_(
                    min=torch.zeros((1, 2), device=x.device),
                    max=torch.tensor([[clamp_shape[1], clamp_shape[0]]], device=x.device))

            proposals_.pred_proj_xy = pred_proj_xy_
            proposals_.pred_z_unscaled = box_z_unscaled_
            proposals_.pred_z_scaled = box_z_scaled_
            proposals_.pred_dims = box_dims_
            proposals_.pred_pose = box_pose_

            # Hack where we move any predicted values from the info (which will be
            # overwritten by the next layer into the current layer's predictions).
            if hasattr(info_, "pred_parameters"):
                proposals_._pred_parameters = info_.pred_parameters
                info_.pred_parameters = None

            if hasattr(info_, "pred_gravity"):
                proposals_._pred_gravity = info_.pred_gravity
                info_.pred_gravity = None

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
        box_2d_pos_embed = self.encoders.box_2d_encoder(torch.stack([
            pred_instances_.pred_boxes.detach() for pred_instances_ in pred_instances
        ], dim=0))

        batch_size, number_queries, _ = box_2d_pos_embed.shape
        box_2d_queries = self.query_embed.weight[None, :number_queries].repeat(batch_size, 1, 1)
        self_attn_mask = torch.zeros((number_queries, number_queries), dtype=torch.bool, device=box_2d_queries.device) # bool

        # Guard during training against too few proposals.
        return EncoderProposals.EncoderPrompt(
            query=box_2d_queries,
            pos_embed=box_2d_pos_embed,
            self_attn_mask=self_attn_mask,
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

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        return output_memory, output_proposals

    def get_proposals(self, memory, mask_flatten, spatial_shapes, sensor):
        output_memory, box_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes, sensor) #sensor无影响

        encoder_proposals = [Instances3D(image_size) for image_size in sensor["image"].data.image_sizes]
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
            topk_proposals_ = torch.topk(encoder_proposal.pred_logits[..., 0], min(top_k, len(encoder_proposal)), dim=0)[1]
            instances_ = encoder_proposal.clone()[topk_proposals_]

            instances.append(instances_)

        # Hack: shove the proposals (needed for encoder supervision) and the topk instances
        #       (which are needed for initialization of queries).
        return (encoder_proposals, instances)

    def inference_single_image(self, output, image_size, topk):            
        # class_prob = output.pred_logits.sigmoid()
        class_prob = F.softmax(output.pred_logits, dim=-1)
        topk_values, topk_indexes = torch.topk(class_prob.view(-1), topk)

        # class_scores = topk_values
        # topk_boxes = topk_indexes // class_prob.shape[-1]
        
        #TODO:changed 25-9-6-lyq
        front_logits = class_prob[..., 1]
        topk_scores, topk_boxes = torch.topk(front_logits.view(-1), topk)
        
        
        labels = topk_indexes % class_prob.shape[-1]
        
        #changed by lyq 25-4-29
        boxes = box_cxcywh_to_xyxy(output.pred_boxes)
        # boxes = output.pred_boxes
        boxes = boxes[topk_boxes]
        xyz = output.pred_xyz[topk_boxes]
        dims = output.pred_dims[topk_boxes]
        pose = output.pred_pose[topk_boxes] #[100,3,3]
        object_desc = output.object_desc[topk_boxes]
        proj_xy = output.pred_proj_xy[topk_boxes]

        #boxes[:,:2] = proj_xy #USE projective cx cy


        result = Instances3D(image_size)
            
        result.scores = topk_scores #class_scores
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

    def inference(self, prompt, output, sensor, topk):
        results = []
        for index, output_ in enumerate(output):
            info = sensor["image"].sensor[index]

            K = info.image.K[-1:].repeat(len(output_), 1, 1)
            output_xyz = torch.bmm(
                torch.linalg.inv(K), torch.cat((output_.pred_z_scaled * output_.pred_proj_xy, output_.pred_z_scaled), dim=-1)[..., None])[..., 0]

            output_.pred_xyz = output_xyz

            if info.has("T_gravity"):
                output_.pred_pose = info.T_gravity[-1:] @ output_.pred_pose
                

            results.append(self.inference_single_image(output_, sensor["image"].data.image_sizes[index], topk=topk))

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
            sensor):

        prompts = []
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
            out.append(
                NestedTensor(x, mask=torch.zeros((x.shape[0], x.shape[-2], x.shape[-1]), dtype=torch.bool, device=x.device)))

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
    


class VGGT_CubifyTransformer(nn.Module):
    """3D目标检测/分割模型的核心模块，整合了特征提取、位置编码、提示机制和解码器"""
    
    def __init__(
        self,
        backbone,       # 特征提取主干网络（如ViT）
        prompting,      # 提示生成机制组件
        decoder,        # 解码器（包含注意力机制和预测头）
        pixel_mean,     # 图像归一化均值
        pixel_std,      # 图像归一化标准差
        pos_embedding,  # 位置编码模块
        vggt_model,  # VGGT模型（用于特征融合）
        fusion_module,  # 特征融合模块
        vggt_merger,  # VGGT特征合并器
        frame_merger, # 帧特征合并器
        depth_model = False,
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
        self.vggt_model = vggt_model  # VGGT模型
        self.fusion_module = fusion_module  # 特征融合模块
        self.vggt_merger = vggt_merger  # VGGT特征合并器
        self.frame_merger = frame_merger  # 帧特征合并器
        self.depth_model = depth_model  # 深度模型（可选）
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
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            
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

    def inference(
        self,
        batched_sensors,  # 批处理的传感器数据字典
        do_preprocess: bool = True,  # 是否执行预处理（保留接口）
        do_postprocess: bool = True,  # 是否执行后处理（保留接口）
    ):
        """推理流程：特征提取→提示生成→解码预测"""
        assert not self.training, "推理必须在eval模式下运行"
        
        img_path = batched_sensors["meta"]['img_path']
        print("img_path:", img_path)
        vggt_image = load_and_preprocess_images(img_path) #(N, 3, H, W) 

        vggt_image = vggt_image.to("cuda")  # 将图像移动到GPU
        # vggt_images = [vggt_image[i] for i in range(vggt_image.shape[0])]
        vggt_images = [vggt_image] #[B, N, 3, H, W]
        
        self.vggt_model.eval()
        
        # 1. 特征提取（主干网络）
        sensor = batched_sensors[self.sensor_name]
        #added but got input of size: [1, 6, 3, 512, 512]
        sensor['image'].data.tensor = sensor['image'].data.tensor.squeeze()
        print("sensor:", sensor['image'].data.tensor.shape)
        if self.depth_model:
            sensor['depth'].data.tensor = sensor['depth'].data.tensor.squeeze()
            print("sensor:", sensor['depth'].data.tensor.shape)
        features = self.backbone(sensor)
        
        # 2. 准备多尺度特征
        srcs, masks, pos_embeds = [], [], []
        for l, (feat, info_) in enumerate(zip(features, sensor["image"].info)):
            # 计算位置编码（融合相机参数）
            pos_embeds_ = self.pos_embedding(feat, sensor)
            pos_embeds.append(pos_embeds_)
            
            # 分离特征和掩码
            src, mask = feat.decompose()
            # 投影到解码器维度
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
        
        # 3. 展平特征（适配Transformer结构）
        flattened = self._flatten(srcs, pos_embeds, masks)
        src_flatten, lvl_pos_embed, mask_flatten, spatial_shapes, level_start_index, valid_ratios = flattened  
        # src_flatten torch.Size([1, 1024, 256])

        # extra processing temporary
        mask_flatten = mask_flatten[:1]  #[1,1024]
        valid_ratios = valid_ratios[:1] #[1,1,2]
        
        # 特征融合
        #extract VGGT增强的3D视觉特征
        vggt_features = self.extract_image_vggt_embeds_3d(vggt_images)  # [batch_size, num_frames, C, H, W]
        print('src_flatten',src_flatten.shape) # [1*N, 1024, 256]
        print('vggt_features',vggt_features.shape)  # [266*N, 2048]
        vggt_features = vggt_features.view(src_flatten.shape[0],-1,vggt_features.shape[-1])  # [1*N, 266, 2048]
        multiframe_fused_features = self.fusion_module(src_flatten, vggt_features)
        print(f"输出尺寸: {multiframe_fused_features.shape}")  
        
        #fuse multiframe features
        fused_features = self.frame_merger(multiframe_fused_features)  # [1*N, 1024, 256] -> [1, 1024, 256]

        # 4. 生成提示（Prompt）
        '''
        Prompter的核心作用是根据图像特征生成​​物体查询（Object Queries）​​，这些查询作为Decoder的输入，承载了模型对场景中“可能存在哪些物体”的初始假设。其本质是一种​​物体候选生成器​​，类似于2D检测中的Region Proposal Network（RPN）
        '''
        prompts = self.prompting.get_image_prompts(
            fused_features, mask_flatten, spatial_shapes, sensor
        )  #modules objects
        prompters = self.prompting.prompters  #Modules structure

        # 5. 解码器处理（核心预测流程）
        '''
        Decoder接收Prompter生成的Object Queries，通过与图像特征的交互（注意力机制）逐步优化这些查询，最终输出精确的3D检测结果
        '''
        _, intermediate_preds = self.decoder(
            fused_features, lvl_pos_embed, mask_flatten, spatial_shapes, 
            level_start_index, valid_ratios, prompts, sensor
        )
        
        # 6. 处理最后层输出
        prompt_outputs = intermediate_preds[-1]
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
                results = prompter.inference(
                    prompt, prompt_outputs_, sensor, 
                    topk=self.topk_per_image
                )
                prompt_start_idx += prompt.number_prompts
                break  # 通常只有一个提示器产生输出
        
        return results

    def forward(self, batched_inputs, do_postprocess=True):
        """训练/推理的统一入口（实际调用推理函数）"""
        return self.inference(batched_inputs)

    def extract_image_vggt_embeds_3d(
        self,
        images_vggt # 输入图像序列 [batch_size, num_frames, C, H, W]
    ):
        """
        提取VGGT增强的3D视觉特征 (image_embeds_3d)
        返回形状: [N, 2048] (N = batch_size * num_frames * h_grid_after_merge * w_grid_after_merge)
        """
        batch_size = len(images_vggt)
        image_embeds_3d = []  # 存储每张图像的3D特征
        
        # 确保VGGT处于评估模式（禁用dropout等训练专用层）
        self.vggt_model.eval()
        
        for i in range(batch_size):
            if images_vggt[i].shape[0] > 0:  # 检查有效图像输入
                n_image = images_vggt[i].shape[0]  # 当前batch的帧数
                height, width = images_vggt[i].shape[-2:]  # 图像原始分辨率
                
                # 获取视觉配置参数
                patch_size = 14 #self.config.vision_config.patch_size
                merge_size = 2 #self.config.vision_config.spatial_merge_size
                
                # 计算分块网格尺寸
                h_grid, w_grid = height // patch_size, width // patch_size
                h_grid_after_merge = h_grid // merge_size
                w_grid_after_merge = w_grid // merge_size
                
                # 自动选择混合精度（根据GPU能力）
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                    # === 核心特征提取流程 ===
                    # 1. 时间翻转（若参考帧非首帧）
                    # if not self.config.reference_frame == "first":
                    #     images_vggt[i] = torch.flip(images_vggt[i], dims=(0))
                    
                    # 2. VGGT特征提取
                    # 使用聚合器提取特征
                    aggregated_tokens_list, patch_start_idx = self.vggt_model.aggregator(images_vggt[i][None])
                    features = aggregated_tokens_list[-2][0, :, patch_start_idx:]
                    
                    # 3. 时间维度校正
                    # if not self.config.reference_frame == "first":
                    #     features = torch.flip(features, dims=(0))
                    
                    # 4. 空间特征重组
                    print("features",features.shape)
                    print('debug',n_image,h_grid, w_grid)
                    features = features.view(n_image, h_grid, w_grid, -1)
                    features = features[:, :h_grid_after_merge * merge_size, 
                                    :w_grid_after_merge * merge_size, :].contiguous()
                    features = features.view(n_image, h_grid_after_merge, merge_size, 
                                        w_grid_after_merge, merge_size, -1)
                    features = features.permute(0, 1, 3, 2, 4, 5).contiguous()
                    
                    # 5. 通过PatchMerger降采样（关键步骤）
                    features = self.vggt_merger(features)#.to(self.visual.dtype))
                    image_embeds_3d.append(features)
        
        # 6. 批量特征拼接
        return torch.cat(image_embeds_3d, dim=0)#.to(self.visual.dtype)

    
def make_vggt_cubify_transformer(vggt_model, dimension, depth_model, embed_dim=256):
    """模型构造工厂函数
    Args:
        dimension: 主干网络特征维度（768/384/192）
        depth_model: 是否使用深度模型
        embed_dim: 解码器嵌入维度（默认256）
    """
    dimension_to_heads = {
        # ViT-B
        768: 12,
        # ViT-S
        384: 6,
        # ViT-T
        192: 3
    }
    
    
    model = VGGT_CubifyTransformer(
        backbone=Joiner(
            backbone=ViT(
                img_size=None,
                patch_size=16,
                embed_dim=dimension,
                depth=12,
                num_heads=dimension_to_heads[dimension],
                window_size=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                window_block_indexes=[
                    0,
                    1,
                    3,
                    4,
                    6,
                    7,
                    9,
                    10,
                ],
                residual_block_indexes=[],
                use_rel_pos=False,
                out_feature="last_feat",
                depth_modality=depth_model,
                depth_window_size=None,
                layer_scale=not depth_model,
                encoder_norm=not depth_model,
                pretrain_img_size=512 if not depth_model else 224
            )),            
        pos_embedding=CameraRayEmbedding(dim=embed_dim),
        prompting=CubifyAnythingPrompting(
            embed_dim=embed_dim,
            prompters=[
                MetricQueries(
                    input_channels=embed_dim,
                    input_stride=16,
                    predictors=None),            
                EncoderProposals(
                    input_channels=embed_dim,
                    input_stride=16,
                    level_strides=[16, 32, 64],
                    predictors=[
                        # Technically, this only gets supervised for 1 class (foreground).
                        ClassPredictor(embed_dim=embed_dim, num_classes=2, num_layers=None),
                        DeltaBox2DPredictor(embed_dim=embed_dim, num_layers=3),
                    ],
                    top_k_test=300,
                ),
            ],
            encoders=PromptEncoders(
                box_2d_encoder=Box2DPromptEncoderLearned(embed_dim=embed_dim)
            )
        ),
        decoder=PromptDecoder(
            embed_dim=embed_dim,
            layer=PreNormGlobalDecoderLayer(
                xattn=GlobalCrossAttention(
                    dim=embed_dim,
                    num_heads=8,
                    rpe_hidden_dim=512,
                    rpe_type="linear",
                    feature_stride=16),
                d_model=embed_dim,
                d_ffn=2048, # for self-attention.
                dropout=0.0,
                activation=F.relu,
                n_heads=8), # for self-attention.
            num_layers=6,
            predictors=[
                ScalePredictor(embed_dim=embed_dim),
                ClassPredictor(embed_dim=embed_dim, num_classes=2, num_layers=None),
                DeltaBox2DPredictor(embed_dim=embed_dim, num_layers=3),
                AbsoluteBox3DPredictor(
                    embed_dim=embed_dim, num_layers=3, pose_type="z", z_type="direct", scale_shift=True)
            ],
            norm=nn.LayerNorm(embed_dim)),
        #specialized for vggt spatial features
        vggt_model=vggt_model,  # VGGT模型（用于特征融合）
        # fusion_module=FeatureFusionModule(fusion_type='add'),
        fusion_module=FeatureFusionModule(fusion_type='add'),
        vggt_merger=VGGTMerger(
                    output_dim=2048, #config.hidden_size, #2048
                    hidden_dim=4096, #getattr(config, "vggt_merger_hidden_dim", 4096), #4096
                    context_dim=2048,
                    spatial_merge_size=2 #config.vision_config.spatial_merge_size, # 2
                ),
        frame_merger=AttentionFusionWithTorch(embed_dim=256, num_heads=8),
        
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        depth_model=depth_model)
    
    return model



        
    
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

