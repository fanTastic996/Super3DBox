import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images, load_and_preprocess_images_resize
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
parser.add_argument('--model_path', type=str, default="/home/lanyuqing/myproject/code/vggt/training/logs/exp001/ckpts/checkpoint_3k_full_17_8imgs.pt", help='Path to model checkpoint')
parser.add_argument('--json_dir', type=str, default="/data1/lyq/scannetpp_json/", help='Input directory path')
parser.add_argument('--data_root', type=str, default="/data1/lyq/scannetpp", help='Input directory path')
parser.add_argument('--save_dir', type=str, default="/data1/lyq/scannetpp_results", help='Output directory path')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing predictions')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


model = VGGT(enable_camera=True, enable_gravity=True, enable_point=False, enable_depth=False, enable_track=False, enable_cubify=True)
_URL = args.model_path
model_dict= torch.load(_URL)
model.load_state_dict(model_dict["model"])

model.eval()
model.to(device) 

json_paths = sorted(os.listdir(args.json_dir))

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
    for frame_ids in all_frame_ids:
        if os.path.exists(f'{args.save_dir}/{scene_id}_{count}_pred.pkl') and os.path.exists(f'{args.save_dir}/pkl/{scene_id}_{count}_pred.pkl') and not args.overwrite:
            count += 1
            print(f'{args.save_dir}/{scene_id}_{count}_pred.pkl already exists')
            continue
        if count > 3:
            continue
        # Load and preprocess example images (replace with your own image paths) 
        # Build image list from indices in all_id_lists[idx]
        image_names = [f"{data_root}/{scene_id}/{count}/{int(i)}.JPG" for i in frame_ids]

        # images = load_and_preprocess_images(image_names).to(device)
        images = load_and_preprocess_images_resize(image_names).to(device)
        print("images", images.shape, torch.max(images), torch.min(images))

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
                # print('extrinsic',extrinsic.shape, extrinsic)
                # print('intrinsic',intrinsic.shape, intrinsic)
                
                save_dict = {}
                save_dict["extrinsics"] = predictions["extrinsic"].cpu().numpy()
                save_dict["intrinsics"] = predictions["intrinsic"].cpu().numpy()
                
                save_dict["box_result"] = {
                    "scores": predictions['pred_scores'][0].cpu().numpy(),
                    "R": predictions['pred_R'][0].cpu().numpy(),
                    "center": predictions['pred_center'][0].cpu().numpy(),
                    "size": predictions['pred_size'][0].cpu().numpy(),
                    # 'ids': predictions['ids'][0].cpu().numpy(),
                    'images': predictions['images'][0].cpu().numpy(),  # (N, 3, H, W)
                    'corners': predictions['pred_corners'][0].cpu().numpy(),
                }
                print("Saving predictions...")
                # 保存到文件
                with open(f'{args.save_dir}/{scene_id}_{count}_pred.pkl', 'wb') as f:
                    pickle.dump(save_dict, f)  # 序列化并写入
                print("saved to ", f'{args.save_dir}/{scene_id}_{count}_pred.pkl')
                # save corners to npy for evaluation
                new_dict = {
                    "scores": save_dict["box_result"]['scores'],
                    "corners": save_dict["box_result"]['corners'],
                }
                if not os.path.exists(f'{args.save_dir}/pkl/'):
                    os.makedirs(f'{args.save_dir}/pkl/')
                with open(f'{args.save_dir}/pkl/{scene_id}_{count}_pred.pkl', 'wb') as f:
                    pickle.dump(new_dict, f)  # 序列化并写入
                print("saved to ", f'{args.save_dir}/pkl/{scene_id}_{count}_pred.pkl')
                
                count += 1