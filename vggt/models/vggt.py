# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead, GravityHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.heads.cubify_head import CubifyHead
from vggt.heads.vggt_cubify_model import *
# from vggt.heads.cubify_head import *
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri, gravity_encoding_to_extri_intri

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_gravity=True, enable_point=True, enable_depth=True, enable_track=True, enable_cubify=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.gravity_head = GravityHead(dim_in=2 * embed_dim) if enable_gravity else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        '''
        Box Head
        '''
        backbone_embedding_dimension=768
        dimension=backbone_embedding_dimension
        dimension_to_heads = {
            # ViT-B
            768: 12,
            # ViT-S
            384: 6,
            # ViT-T
            192: 3
        }
        cubify_embed_dim = 256
        depth_model = False
        self.box_head = CubifyHead(
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
                    depth_window_size=None,
                    layer_scale=not depth_model,
                    encoder_norm=not depth_model,
                    pretrain_img_size=512 if not depth_model else 224
                )),            
            pos_embedding=CameraRayEmbedding(dim=cubify_embed_dim),
            prompting=CubifyAnythingPrompting(
                embed_dim=cubify_embed_dim,
                prompters=[
                    MetricQueries(
                        input_channels=cubify_embed_dim,
                        input_stride=16,
                        predictors=None),            
                    EncoderProposals(
                        input_channels=cubify_embed_dim,
                        input_stride=16,
                        level_strides=[16, 32, 64],
                        predictors=[
                            # Technically, this only gets supervised for 1 class (foreground).
                            ClassPredictor(embed_dim=cubify_embed_dim, num_classes=2, num_layers=None),
                            DeltaBox2DPredictor(embed_dim=cubify_embed_dim, num_layers=3),
                        ],
                        top_k_test=50 #100, #300,
                    ),
                ],
                encoders=PromptEncoders(
                    box_2d_encoder=Box2DPromptEncoderLearned(embed_dim=cubify_embed_dim)
                )
            ),
            decoder=PromptDecoder(
                embed_dim=cubify_embed_dim,
                #几何感知的 Transformer 解码层
                # Self-Attention：让 queries 之间通信；
                # GlobalCrossAttention：让每个 query 聚焦自己 2D 框对应的图像特征；
                 #FFN：非线性特征增强；
                # LayerNorm + 残差：稳定训练。
                # 堆叠 6 层后构成整个 Decoder，用于多轮 refinement（从粗框到精框，从 2D 到 3D）。
                layer=PreNormGlobalDecoderLayer(
                    # 这个模块是用相对位置偏置 (Relative Position Encoding, RPE) 来计算注意力，使得 cross-attention 能考虑 query 的 2D 投影与图像特征位置的关系。它本质上就是 DETR 的跨注意力，但带有 box-aware 位置引导。
                    xattn=GlobalCrossAttention( 
                        dim=cubify_embed_dim,
                        num_heads=8,
                        rpe_hidden_dim=512,
                        rpe_type="linear",
                        feature_stride=16),
                    d_model=cubify_embed_dim,
                    d_ffn=2048, # for self-attention.
                    dropout=0.0,
                    activation=F.relu,
                    n_heads=8), # for self-attention.
                num_layers=6,
                predictors=[
                    ScalePredictor(embed_dim=cubify_embed_dim),
                    ClassPredictor(embed_dim=cubify_embed_dim, num_classes=2, num_layers=None),
                    DeltaBox2DPredictor(embed_dim=cubify_embed_dim, num_layers=3),
                    AbsoluteBox3DPredictor(
                        embed_dim=cubify_embed_dim, num_layers=3, pose_type="z", z_type="direct", scale_shift=True)
                ],
                norm=nn.LayerNorm(cubify_embed_dim)),
            #specialized for vggt spatial features
            # fusion_module=FeatureFusionModule_v2(in_channels=2048,
            #                                  out_channels=256,
            #                                  num_heads=8,
            #                                  dropout=0.1,
            #                                  fusion_type='add'),
            # fusion_module=FeatureFusionModule_v3(d_clip=256,
            #                                  d_spatial_encoder=2048,
            #                                  d_attn=256,
            #                                  num_heads=8),
            
            # vggt_merger=VGGTMerger(
            #             output_dim=2048, #config.hidden_size, #2048
            #             hidden_dim=4096, #getattr(config, "vggt_merger_hidden_dim", 4096), #4096
            #             context_dim=2048,
            #             spatial_merge_size=2 #config.vision_config.spatial_merge_size, # 2
            #         ),
            # frame_merger=AttentionFusionWithTorch(embed_dim=256, num_heads=8),
            # frame_merger=LightweightCrossViewFusion(feat_dim=256),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
            topk_per_image=50) if enable_cubify else None
            #TODO: change 100->50
            
    def forward(self, images: torch.Tensor, intrinsics: torch.Tensor= None, extrinsics: torch.Tensor= None, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        # print("aggregated_tokens_list",len(aggregated_tokens_list),aggregated_tokens_list[0].shape)
        # aggregated_tokens_list 24 torch.Size([4, 3, 1263, 2048])
        predictions = {}

        # with torch.cuda.amp.autocast(enabled=False):
        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
            
            if self.gravity_head is not None:
                gravity_enc_list = self.gravity_head(aggregated_tokens_list)
                predictions["gravity_enc"] = gravity_enc_list[-1]  # pose encoding of the last iteration
                predictions["gravity_enc_list"] = gravity_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.box_head is not None:
                # [Seq, N_img, ...]
                # extract ex and intrinsics [Seq, N, 3, 4/3]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
                
                gravity_init = gravity_encoding_to_extri_intri(predictions["gravity_enc"], images.shape[-2:]) #[B,N,3,4] every single frame has a pred gravity
                gravity = gravity_init[:,:,:3,:3] #[B,1,3,3]
                
                # print("input",images.shape) #([4(B), 3(N_img), 3, 476, 518])
                box_result = self.box_head(
                # all_corners, all_logits = self.box_head(
                    images,
                    aggregated_tokens_list,
                    patch_start_idx,
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    gravity=gravity
                    # images=images,
                )
            
                # predictions["box_result"] = box_result
                # predictions["pred_corners"] = all_corners
                # predictions["pred_logits"] = all_logits
                
                predictions["pred_corners"] = [box_result[batch_idx].pred_boxes_3d.corners for batch_idx in range(len(box_result))]
                predictions["pred_logits"] = [box_result[batch_idx].pred_logits for batch_idx in range(len(box_result))]
                
                predictions["pred_scores"] = [box_result[batch_idx].scores for batch_idx in range(len(box_result))]
                predictions["pred_R"] = [box_result[batch_idx].pred_boxes_3d.R for batch_idx in range(len(box_result))]
                predictions["pred_center"] = [box_result[batch_idx].pred_boxes_3d.gravity_center for batch_idx in range(len(box_result))]
                predictions["pred_size"] = [box_result[batch_idx].pred_boxes_3d.dims for batch_idx in range(len(box_result))]
                
                predictions["extrinsics"] = extrinsic
                predictions["intrinsics"] = intrinsic
                
                
        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        # if not self.training:
            # predictions["images"] = images  # store the images for visualization during inference
        predictions["images"] = images
        
        return predictions

if __name__ == "__main__":
    '''
    Box Head
    '''
    backbone_embedding_dimension=768
    dimension=backbone_embedding_dimension
    dimension_to_heads = {
        # ViT-B
        768: 12,
        # ViT-S
        384: 6,
        # ViT-T
        192: 3
    }
    cubify_embed_dim = 256
    depth_model = False
    box_head = CubifyHead(
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
                depth_window_size=None,
                layer_scale=not depth_model,
                encoder_norm=not depth_model,
                pretrain_img_size=512 if not depth_model else 224
            )),            
        pos_embedding=CameraRayEmbedding(dim=cubify_embed_dim),
        prompting=CubifyAnythingPrompting(
            embed_dim=cubify_embed_dim,
            prompters=[
                MetricQueries(
                    input_channels=cubify_embed_dim,
                    input_stride=16,
                    predictors=None),            
                EncoderProposals(
                    input_channels=cubify_embed_dim,
                    input_stride=16,
                    level_strides=[16, 32, 64],
                    predictors=[
                        # Technically, this only gets supervised for 1 class (foreground).
                        ClassPredictor(embed_dim=cubify_embed_dim, num_classes=2, num_layers=None),
                        DeltaBox2DPredictor(embed_dim=cubify_embed_dim, num_layers=3),
                    ],
                    top_k_test=300,
                ),
            ],
            encoders=PromptEncoders(
                box_2d_encoder=Box2DPromptEncoderLearned(embed_dim=cubify_embed_dim)
            )
        ),
        decoder=PromptDecoder(
            embed_dim=cubify_embed_dim,
            layer=PreNormGlobalDecoderLayer(
                xattn=GlobalCrossAttention(
                    dim=cubify_embed_dim,
                    num_heads=8,
                    rpe_hidden_dim=512,
                    rpe_type="linear",
                    feature_stride=16),
                d_model=cubify_embed_dim,
                d_ffn=2048, # for self-attention.
                dropout=0.0,
                activation=F.relu,
                n_heads=8), # for self-attention.
            num_layers=6,
            predictors=[
                ScalePredictor(embed_dim=cubify_embed_dim),
                ClassPredictor(embed_dim=cubify_embed_dim, num_classes=2, num_layers=None),
                DeltaBox2DPredictor(embed_dim=cubify_embed_dim, num_layers=3),
                AbsoluteBox3DPredictor(
                    embed_dim=cubify_embed_dim, num_layers=3, pose_type="z", z_type="direct", scale_shift=True)
            ],
            norm=nn.LayerNorm(cubify_embed_dim)),
        #specialized for vggt spatial features
        fusion_module=FeatureFusionModule_v2(in_channels=2048,
                                             out_channels=256,
                                             num_heads=8,
                                             dropout=0.1,
                                             fusion_type='add'),
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
    print(box_head)