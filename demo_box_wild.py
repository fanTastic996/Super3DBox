import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import *
from visual_util import segment_sky, download_file_from_url
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import pickle
import os
from PIL import Image




device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT(enable_camera=True, enable_gravity=True, enable_point=False, enable_depth=True, enable_track=False, enable_cubify=True,enable_depth_modality=False)
# model = VGGT(enable_camera=True, enable_gravity=False, enable_point=False, enable_depth=False, enable_track=False, enable_cubify=True)
# _URL = "/data1/lyq/logs/exp001/ckpts/checkpoint_3k_new_mvf_depthmode_33epoch.pt"
# _URL = "/data1/lyq/logs/exp001/ckpts/checkpoint_3k_new_mvf_rgbmode_40epoch_Grefine.pt"
_URL = "/data1/lyq/logs/exp001/ckpts/checkpoint_3k_new_mvf_rgbmode_40epoch.pt"
# _URL = "/data1/lyq/logs/exp001/ckpts/checkpoint_3k_new_mvf_depthmode_33epoch.pt"
# _URL = "/data1/lyq/logs/exp001/ckpts/checkpoint_250125_ca1m_depth_nomvf_5epoch_depthmode.pt"

# _URL = "/data1/lyq/logs/exp001/ckpts/checkpoint_3k_full_144.pt"
# _URL = "/home/lanyuqing/myproject/vggt/training/logs/exp001/ckpts/checkpoint_45444750_180_200_gravity_query.pt"
model_dict= torch.load(_URL)
model.load_state_dict(model_dict["model"]) 


model.eval()
model.to(device)

# data_root = '/data1/lyq/scannetpp'
# scene_id = '0a76e06478'
# scene_id = '1ae9e5d2a6'
# data_root = '/data1/lyq/scannetpp_iphone/'
# data_root = '/data1/lyq/CA1M-dataset/CA1M-dataset/test'
# scene_id = '21d970d8de' #'101_2' #'036bce3393'#'fr1_desk' #'building2' #'room0' #'scene0000' #'47895364'
# scene_id = '42897561' #'42898486'

data_root = '/data1/lyq/inthewild/scenenn005'

for j in range(10):
    scene_id = f'sample_0{j}'
    # scene_id = 'tank0'
    # scene_id = 'scene0000_00'


    # scene_id = 'scacsd' #'47895364'
    # scene_id = '42444750' #'47332808'
    # Load and preprocess example images (replace with your own image paths) 
    # scene_id = '47334115'
    # image_names = [f"{data_root}/{scene_id}/rgb/83.png", f"{data_root}/{scene_id}/rgb/93.png", f"{data_root}/{scene_id}/rgb/103.png",  f"{data_root}/{scene_id}/rgb/113.png"]  
    # image_names = ["/data1/lyq/100.jpg", "/data1/lyq/120.jpg", "/data1/lyq/140.jpg"]  
    # image_names = ["/data1/lyq/11.png", "/data1/lyq/21.png","/data1/lyq/31.png"]  

    # image_names = [f"{data_root}/{scene_id}/rgb/277.png", f"{data_root}/{scene_id}/rgb/287.png"]
    # image_names = [f"{data_root}/1000.jpg", f"{data_root}/2000.jpg", f"{data_root}/3000.jpg"]
    # image_names = [f"{data_root}/5000.jpg", f"{data_root}/6000.jpg", f"{data_root}/7000.jpg",f"{data_root}/8000.jpg"]
    # image_names = [f"{data_root}/{scene_id}/2/400.jpg", f"{data_root}/{scene_id}/2/555.jpg", f"{data_root}/{scene_id}/2/565.jpg", f"{data_root}/{scene_id}/2/635.jpg", f"{data_root}/{scene_id}/2/745.jpg"]
    # image_names = [f"{data_root}/{scene_id}/rgb/3.png", f"{data_root}/{scene_id}/rgb/33.png"]

    # image_names = [f"{data_root}/{scene_id}/rgb/75.png", f"{data_root}/{scene_id}/rgb/115.png",f"{data_root}/{scene_id}/rgb/135.png",f"{data_root}/{scene_id}/rgb/155.png",f"{data_root}/{scene_id}/rgb/195.png",f"{data_root}/{scene_id}/rgb/205.png",f"{data_root}/{scene_id}/rgb/285.png",f"{data_root}/{scene_id}/rgb/310.png"]

    # Replica-room0
    image_names = [name for name in os.listdir(f"{data_root}/{scene_id}") if os.path.isfile(os.path.join(data_root, scene_id, name))]
    # 按照文件名 '.' 之前的数字排序
    def num_prefix(name):
        return int(name.split('.')[0]) if name.split('.')[0].isdigit() else float('inf')
    image_names = sorted(image_names, key=num_prefix)
    # 拼接完整路径
    image_names = [f"{data_root}/{scene_id}/{name}" for name in image_names]

    # image_names = [f"{data_root}/{scene_id}/0.jpg", f"{data_root}/{scene_id}/190.jpg",f"{data_root}/{scene_id}/622.jpg",f"{data_root}/{scene_id}/790.jpg",f"{data_root}/{scene_id}/954.jpg",f"{data_root}/{scene_id}/1880.jpg"]

    # image_names = [f"{data_root}/{scene_id}/rgb/185.png", f"{data_root}/{scene_id}/rgb/230.png", f"{data_root}/{scene_id}/rgb/455.png"]
    # image_names = [f"{data_root}/{scene_id}/1/31.JPG", f"{data_root}/{scene_id}/1/36.JPG",f"{data_root}/{scene_id}/1/41.JPG",f"{data_root}/{scene_id}/1/46.JPG"]
    # image_names = [f"{data_root}/{scene_id}/2/140.JPG", f"{data_root}/{scene_id}/2/145.JPG"]

    # image_names = [f"{data_root}/{scene_id}/2/140.JPG", f"{data_root}/{scene_id}/2/145.JPG",f"{data_root}/{scene_id}/2/150.JPG",f"{data_root}/{scene_id}/2/155.JPG",f"{data_root}/{scene_id}/2/160.JPG",f"{data_root}/{scene_id}/2/165.JPG",f"{data_root}/{scene_id}/2/170.JPG",f"{data_root}/{scene_id}/2/175.JPG"]
    # image_names = [f"{data_root}/{scene_id}/1/306.JPG"]

    # image_names = [f"{data_root}/{scene_id}/rgb/2.png", f"{data_root}/{scene_id}/rgb/219.png"]
    # image_names = [f"{data_root}/{scene_id}/rgb/0.png"] #219

    # image_names = [f"{data_root}/{scene_id}/rgb/2.png", f"{data_root}/{scene_id}/rgb/219.png", f"{data_root}/{scene_id}/rgb/436.png", f"{data_root}/{scene_id}/rgb/653.png", f"{data_root}/{scene_id}/rgb/870.png",f"{data_root}/{scene_id}/rgb/1087.png",f"{data_root}/{scene_id}/rgb/1304.png",f"{data_root}/{scene_id}/rgb/1521.png"]


    # image_names = [f"{data_root}/{scene_id}/rgb/200.png"]  
    # images = load_and_preprocess_images_resize(image_names).to(device)

    images = load_and_preprocess_images_original(image_names).to(device)
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
            # print('extrinsic',extrinsic.shape, extrinsic)
            # print('intrinsic',intrinsic.shape, intrinsic)

            
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            depth_map = predictions["depth"][0]
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
            init_threshold_val = np.percentile(conf_flat, 25.0)
            init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
            my_points = points[init_conf_mask]   # (N,3)
            my_colors = colors_flat[init_conf_mask]       # (N,3)
            out_path = f'./vis_results/{scene_id}_scene.ply' # <- 改成你指定路径
            save_colored_pointcloud(my_points, my_colors, out_path)
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            
            
            
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