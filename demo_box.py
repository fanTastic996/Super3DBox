import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images, load_and_preprocess_images_mine, load_and_preprocess_images_resize
from visual_util import segment_sky, download_file_from_url
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import pickle
import os
from PIL import Image




device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT(enable_camera=True, enable_gravity=True, enable_point=False, enable_depth=True, enable_track=False, enable_cubify=True)
# model = VGGT(enable_camera=True, enable_gravity=False, enable_point=False, enable_depth=False, enable_track=False, enable_cubify=True)
# _URL = "/data1/lyq/logs/exp001/ckpts/checkpoint_250125_ca1m_depth_nomvf_5epoch.pt"
_URL = "/data1/lyq/logs/exp001/ckpts/checkpoint.pt"
# _URL = "/data1/lyq/logs/exp001/ckpts/checkpoint_3k_full_144.pt"
# _URL = "/home/lanyuqing/myproject/vggt/training/logs/exp001/ckpts/checkpoint_45444750_180_200_gravity_query.pt"
model_dict= torch.load(_URL)
model.load_state_dict(model_dict["model"]) 


model.eval()
model.to(device)

# data_root = '/data1/lyq/scannetpp'
# scene_id = '0a76e06478'
# data_root = '/data1/lyq'
data_root = '/data1/lyq/CA1M-dataset/CA1M-dataset/test'
# scene_id = '42899112' #'47895364'
scene_id = '42897538'

# scene_id = 'scacsd' #'47895364'
# scene_id = '42444750' #'47332808'
# Load and preprocess example images (replace with your own image paths) 
# scene_id = '47334115'
# image_names = [f"{data_root}/{scene_id}/rgb/83.png", f"{data_root}/{scene_id}/rgb/93.png", f"{data_root}/{scene_id}/rgb/103.png",  f"{data_root}/{scene_id}/rgb/113.png"]  
# image_names = ["/data1/lyq/70.JPG", "/data1/lyq/80.JPG", "/data1/lyq/85.JPG"]  

# image_names = [f"{data_root}/{scene_id}/rgb/277.png", f"{data_root}/{scene_id}/rgb/287.png"]
# image_names = [f"{data_root}/0.jpg", f"{data_root}/110.jpg", f"{data_root}/420.jpg"]
# image_names = [f"{data_root}/{scene_id}/rgb/50.png", f"{data_root}/{scene_id}/rgb/70.png"]
image_names = [f"{data_root}/{scene_id}/rgb/620.png", f"{data_root}/{scene_id}/rgb/729.png", f"{data_root}/{scene_id}/rgb/810.png"]
# image_names = [f"{data_root}/{scene_id}/1/31.JPG", f"{data_root}/{scene_id}/1/36.JPG",f"{data_root}/{scene_id}/1/41.JPG",f"{data_root}/{scene_id}/1/46.JPG"]

# image_names = [f"{data_root}/{scene_id}/rgb/2.png", f"{data_root}/{scene_id}/rgb/219.png"]
# image_names = [f"{data_root}/{scene_id}/rgb/0.png"] #219

# image_names = [f"{data_root}/{scene_id}/rgb/2.png", f"{data_root}/{scene_id}/rgb/219.png", f"{data_root}/{scene_id}/rgb/436.png", f"{data_root}/{scene_id}/rgb/653.png", f"{data_root}/{scene_id}/rgb/870.png",f"{data_root}/{scene_id}/rgb/1087.png",f"{data_root}/{scene_id}/rgb/1304.png",f"{data_root}/{scene_id}/rgb/1521.png"]


# image_names = [f"{data_root}/{scene_id}/rgb/200.png"]  
images = load_and_preprocess_images_resize(image_names).to(device)
# images = load_and_preprocess_images_mine(image_names).to(device)
# images = load_and_preprocess_images(image_names).to(device)
print("images", images.shape, torch.max(images), torch.min(images))

wanted_keys = ["extrinsic", "intrinsic", "box_result"]

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images, inference_tag=True)
        
        print("predictions", predictions.keys())
        
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        print('extrinsic',extrinsic.shape, extrinsic)
        print('intrinsic',intrinsic.shape, intrinsic)
        # pred_3d_boxes = predictions['box_result'][0]
  
        
        save_dict = {}
        save_dict["extrinsics"] = predictions["extrinsic"].cpu().numpy()
        save_dict["intrinsics"] = predictions["intrinsic"].cpu().numpy()
        # bboxes_3d = pred_3d_boxes.pred_boxes_3d
        # save_dict["box_result"] = {
        #     "scores": pred_3d_boxes.scores.cpu().numpy(),
        #     "R": bboxes_3d.R.cpu().numpy(),
        #     "center": bboxes_3d.gravity_center.cpu().numpy(),
        #     "size": bboxes_3d.dims.cpu().numpy(),
        # }
        
        save_dict["box_result"] = {
            "scores": predictions['pred_scores'][0].cpu().numpy(),
            "R": predictions['pred_R_g'][0].cpu().numpy(),
            "center": predictions['pred_center_g'][0].cpu().numpy(),
            "size": predictions['pred_size_g'][0].cpu().numpy(),
            # 'ids': predictions['ids'][0].cpu().numpy(),
            'images': predictions['images'][0].cpu().numpy(),  # (N, 3, H, W)
            'corners': predictions['pred_corners_g'][0].cpu().numpy(),
            # 'corners_g': predictions['pred_corners_g'][0].cpu().numpy(),
            # 'center_g': predictions['pred_center_g'][0].cpu().numpy(),
        }
        
        # valid_mask = pred_3d_boxes.scores.cpu().numpy()>=0.0
        # print("center", bboxes_3d.gravity_center.cpu().numpy()[valid_mask])
        # print("size", bboxes_3d.dims.cpu().numpy()[valid_mask])
        # print("R", bboxes_3d.R.cpu().numpy()[valid_mask])
        print("scores", predictions['pred_scores'][0].cpu().numpy())
        # print(boxes_3d.R.cpu().numpy().shape,poses.shape)
        print("Saving predictions...")
        # 保存到文件
        with open(f'./vis_results/{scene_id}_pred.pkl', 'wb') as f:
            pickle.dump(save_dict, f)  # 序列化并写入
        print("saved to ", f'./vis_results/{scene_id}_pred.pkl')