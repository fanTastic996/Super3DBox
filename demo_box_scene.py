import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from visual_util import segment_sky, download_file_from_url
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import pickle
import os
from PIL import Image

def get_inferece_list(seq_name,interval = 20,data_store = 710):
    import numpy as np

    # 固定随机种子，保证可复现
    np.random.seed(42)

    # 计算起始索引最大值
    all_id_lists = []
    for _ in range(20):
        img_per_seq = np.random.randint(2, 4)
        max_start = data_store - (img_per_seq - 1) * interval
        start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
        ids = np.array([start_idx + i * interval for i in range(img_per_seq)])
        all_id_lists.append(ids)

    print(all_id_lists)
    return all_id_lists

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


model = VGGT(enable_camera=True, enable_gravity=True, enable_point=False, enable_depth=False, enable_track=False, enable_cubify=True)
_URL = "/home/lanyuqing/myproject/VGGT/training/logs/exp001/ckpts/checkpoint.pt"
model_dict= torch.load(_URL)
model.load_state_dict(model_dict["model"])


model.eval()
model.to(device)

data_root = '/home/lanyuqing/dataset/training'
scene_id = '45261413' #43828340:635 42444750:711 45261413:406



all_id_lists = get_inferece_list(scene_id, interval=10, data_store=406)


for idx in range(len(all_id_lists)):
    # Load and preprocess example images (replace with your own image paths) 
    # Build image list from indices in all_id_lists[idx]
    image_names = [f"{data_root}/{scene_id}/rgb/{int(i)}.png" for i in all_id_lists[idx]]


    images = load_and_preprocess_images(image_names).to(device)
    print("images", images.shape, torch.max(images), torch.min(images))

    wanted_keys = ["extrinsic", "intrinsic", "box_result"]

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
            
            print("predictions", predictions.keys())
            
            print("Converting pose encoding to extrinsic and intrinsic matrices...")
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            print('extrinsic',extrinsic.shape, extrinsic)
            print('intrinsic',intrinsic.shape, intrinsic)
            
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
            }
            print("Saving predictions...")
            # 保存到文件
            with open(f'./vis_results/{scene_id}_pred_{idx}.pkl', 'wb') as f:
                pickle.dump(save_dict, f)  # 序列化并写入
            print("saved to ", f'./vis_results/{scene_id}_pred_{idx}.pkl')