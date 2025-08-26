import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from visual_util import segment_sky, download_file_from_url
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import pickle




device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


model = VGGT(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False, enable_cubify=True)
_URL = "/home/lanyuqing/myproject/vggt/training/logs/exp001/ckpts/checkpoint.pt"
model_dict= torch.load(_URL)
model.load_state_dict(model_dict["model"])
model.eval()
model.to(device)

data_root = '/data/lyq/ca1m/ca1m/train-CA-1M-slam'
# Load and preprocess example images (replace with your own image paths) 
# scene_id = '43649409'
# image_names = [f"{data_root}/{scene_id}/rgb/478.png", f"{data_root}/{scene_id}/rgb/546.png"]  
scene_id = '42444750'
image_names = [f"{data_root}/{scene_id}/rgb/180.png", f"{data_root}/{scene_id}/rgb/200.png"]  
images = load_and_preprocess_images(image_names).to(device)

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
        print('extrinsic',extrinsic.shape)
        print('intrinsic',intrinsic.shape)
        pred_3d_boxes = predictions['box_result'][0]
  
        
        save_dict = {}
        save_dict["extrinsic"] = predictions["extrinsic"].cpu().numpy()
        save_dict["intrinsic"] = predictions["intrinsic"].cpu().numpy()
        bboxes_3d = pred_3d_boxes.pred_boxes_3d
        save_dict["box_result"] = {
            "scores": pred_3d_boxes.scores.cpu().numpy(),
            "R": bboxes_3d.R.cpu().numpy(),
            "center": bboxes_3d.gravity_center.cpu().numpy(),
            "size": bboxes_3d.dims.cpu().numpy(),
        }
        print("center", bboxes_3d.gravity_center.cpu().numpy())
        
        # print(boxes_3d.R.cpu().numpy().shape,poses.shape)
        print("Saving predictions...")
        # 保存到文件
        with open(f'./vis_results/{scene_id}_pred.pkl', 'wb') as f:
            pickle.dump(save_dict, f)  # 序列化并写入
        print("saved to ", f'./vis_results/{scene_id}_pred.pkl')