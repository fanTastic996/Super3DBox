import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import *
from visual_util import segment_sky, download_file_from_url
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import pickle
import os
from PIL import Image
import argparse
import json
import numpy as np
parser = argparse.ArgumentParser(description="IGGT Scene Processor")
parser.add_argument('--model_path', type=str, default="/home/lanyuqing/myproject/code/vggt/training/logs/exp001/ckpts/checkpoint_3k_new_mvf_rgbmode_40epoch.pt", help='Path to model checkpoint')
parser.add_argument('--json_dir', type=str, default="/data1/lyq/scannetpp-benchmark/", help='Input directory path')
parser.add_argument('--data_root', type=str, default="/data1/lyq/scannetpp_iphone", help='Input directory path')
parser.add_argument('--save_dir', type=str, default="/data1/lyq/scannetpp_results_selfz", help='Output directory path')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing predictions')
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


model = VGGT(enable_camera=True, enable_gravity=True, enable_point=False, enable_depth=True, enable_track=False, enable_cubify=True, enable_depth_modality=False)
_URL = args.model_path
model_dict= torch.load(_URL)
model.load_state_dict(model_dict["model"])

model.eval()
model.to(device) 

json_paths = sorted(os.listdir(args.json_dir))


good_list = [
        "28a9ee4557", "286b55a2bf", "260fa55d50", "21d970d8de", "210f741378",
        "1d003b07bd", "1b75758486", "1ae9e5d2a6", "1841a0b525", "16c9bd2e1e",
        "1204e08f17", "116456116b", "0b031f3119", "09bced689e", "08bbbdcc3d",
        "07ff1c45bb", "079a326597", "0a7cc12c0e", "0a76e06478", "0a184cf634"
    ]

for json_path in json_paths:
    json_path = os.path.join(args.json_dir, json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 10 组 window
    windows = data["windows"]  # list, len == 10
    all_frame_ids = [w["frame_ids"] for w in windows]  # List[List[int]]
    scene_id = data["seq_dir"].split("/")[-1]
    data_root = args.data_root
    count = 0
    
    if scene_id not in good_list:
        print(f'{scene_id} not in good_list')
        continue
    
    for frame_ids in all_frame_ids:
        # if os.path.exists(f'{args.save_dir}/{scene_id}_{count}_pred.pkl') and os.path.exists(f'{args.save_dir}/pkl/{scene_id}_{count}_pred.pkl') and not args.overwrite:
        #     count += 1
        #     print(f'{args.save_dir}/{scene_id}_{count}_pred.pkl already exists')
        #     continue
        # if count > 3:
        #     continue
        # Load and preprocess example images (replace with your own image paths) 
        # Build image list from indices in all_id_lists[idx]
        image_names = [f"{data_root}/{scene_id}/{count}/{int(i)}.jpg" for i in frame_ids]
        depth_path_list = [f"{data_root}/{scene_id}/{count}/{int(i)}.png" for i in frame_ids]
        
        # images = load_and_preprocess_images(image_names).to(device)
        images = load_and_preprocess_images_original(image_names).to(device)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        K=np.loadtxt(f"{data_root}/{scene_id}/K.txt").reshape(3,3)
        all_poses = np.load(f"{data_root}/{scene_id}/all_poses.npy").reshape(-1, 4, 4)
        # gt_data = load_gt_data(f"{data_root}/{scene_id}", depth_path_list)
        depths, valid_mask, Ks_new, gt_world_points = load_and_preprocess_depths_and_valid_mask(image_names, depth_path_list, K, all_poses[frame_ids])
        depths = depths.squeeze(1).detach().cpu().numpy()
        valid_mask = valid_mask.squeeze(1).detach().cpu().numpy()
        # print("depths", depths.shape)
        # print("valid_mask", valid_mask.shape)
        # print("Ks_new", Ks_new.shape)
        # print("gt_world_points", gt_world_points.shape)
        # exit(0)
        save_gt_data(gt_world_points, valid_mask, depths, os.path.join(args.save_dir, "gt_data",scene_id, str(count)))
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        wanted_keys = ["extrinsic", "intrinsic", "box_result"]

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                predictions = model(images, inference_tag=True)
                
                # print("predictions", predictions.keys())
                
                print("Converting pose encoding to extrinsic and intrinsic matrices...")
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
                predictions["extrinsic"] = extrinsic
                predictions["intrinsic"] = intrinsic

                depth_map = predictions["depth"][0]
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                depth_conf = predictions["depth_conf"][0].detach().cpu().numpy()  # (S, H, W)
                world_points = unproject_depth_map_to_point_map(
                    depth_map, predictions["extrinsic"][0], predictions["intrinsic"][0]
                )
                conf = depth_conf
                #save vggt pointmap to ply based on filtering
                colors = images.detach().cpu().numpy().transpose(0, 2, 3, 1)  # now (S, H, W, 3)
                # Flatten
                points = world_points.reshape(-1, 3)
                colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)

                conf_flat = conf.reshape(-1)  
                # Create the main point cloud handle
                # Compute the threshold value as the given percentile
                init_threshold_val = np.percentile(conf_flat, args.conf_threshold)
                init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
                my_points = points[init_conf_mask]   # (N,3)
                my_colors = colors_flat[init_conf_mask]       # (N,3)
                os.makedirs(os.path.join(args.save_dir, "ply"), exist_ok=True)
                out_path = os.path.join(args.save_dir, "ply", f'{scene_id}_{count}_pred.ply') # <- 改成你指定路径
                save_colored_pointcloud(my_points, my_colors, out_path)
                

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                
                
                save_dict = {}
                save_dict["extrinsics"] = predictions["extrinsic"][0].cpu().numpy()
                save_dict["intrinsics"] = predictions["intrinsic"][0].cpu().numpy()
                save_dict["world_points"] = world_points
                
                save_dict["box_result"] = {
                    "scores": predictions['pred_scores'][0].cpu().numpy(),
                    "R": predictions['pred_R_g'][0].cpu().numpy(),
                    "center": predictions['pred_center_g'][0].cpu().numpy(),
                    "size": predictions['pred_size_g'][0].cpu().numpy(),
                    # 'ids': predictions['ids'][0].cpu().numpy(),
                    'images': predictions['images'][0].cpu().numpy(),  # (N, 3, H, W)
                    'corners': predictions['pred_corners_g'][0].cpu().numpy(),
                }
                print("Saving predictions...")
                with open(f'{args.save_dir}/{scene_id}_{count}_pred.pkl', 'wb') as f:
                    pickle.dump(save_dict, f)  
                print("saved to ", f'{args.save_dir}/{scene_id}_{count}_pred.pkl')
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # save corners to npy for evaluation
                new_dict = {
                    "scores": save_dict["box_result"]['scores'],
                    "corners": save_dict["box_result"]['corners'],
                    "size": save_dict["box_result"]['size'],
                    "center": save_dict["box_result"]['center'],
                    "R": save_dict["box_result"]['R'],
                }
                #TODO:后面align的时候同步修改corners和centers，然后写回pkl
                os.makedirs(f'{args.save_dir}/pkl/', exist_ok=True)
                with open(f'{args.save_dir}/pkl/{scene_id}_{count}_pred.pkl', 'wb') as f:
                    pickle.dump(new_dict, f)  # 序列化并写入
                print("saved to ", f'{args.save_dir}/pkl/{scene_id}_{count}_pred.pkl')
                count += 1