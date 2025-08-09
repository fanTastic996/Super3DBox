import os
import numpy as np
import json
import cv2
import shutil
import open3d as o3d
import copy

root = '/data/CA-1M-unzip'
target_dir = '/data/CA-1M-slam'
dir_path = sorted(os.listdir(root))
# print("dirpath",dir_path)
number_dir = sorted([i.split('-')[-1] for i in dir_path])

print("dirpath", dir_path)

h_seqs = []
v_seqs = []
complete_h_seqs = []
complete_v_seqs = []

for seq in dir_path:
    sub_seq = seq.split('-')[-1]
    # print("sub_seq", sub_seq)
    # print("seq",seq)
    # rgb_dir = os.path.join(target_dir,seq,sub_seq,"rgb")
    rgb_dir = os.path.join(root, seq, sub_seq)
    image_dir = sorted(os.listdir(rgb_dir))
    # print("image_dir", image_dir)
    num_images = len(image_dir)
    # print(num_images)
    first_h = 0
    first_w = 0
    h_count = 0
    v_count = 0
    for frame_id in range(int((num_images - 1) / 4)):

        # t_rgb = os.path.join(target_dir,seq,"rgb",str(frame_id)+".png")
        # t_depth = os.path.join(target_dir,seq,"depth",str(frame_id)+".png")
        t_rgb = os.path.join(root, seq, sub_seq, image_dir[frame_id * 4 + 3], "image.png")
        t_depth = os.path.join(root, seq, sub_seq, image_dir[frame_id * 4 + 2], "depth.png")

        cur_d = cv2.imread(t_depth, -1)

        if cur_d.shape[0] > cur_d.shape[1]:
            v_count += 1
        else:
            h_count += 1

    if v_count > h_count:
        v_seqs.append(sub_seq)
        if h_count == 0:
            complete_v_seqs.append(sub_seq)
        print(f"垂直序列: {sub_seq} - 垂直图像数量: {v_count}, 水平图像数量: {h_count}")
    else:
        h_seqs.append(sub_seq)
        if v_count == 0:
            complete_h_seqs.append(sub_seq)
        print(f"水平序列: {sub_seq} - 水平图像数量: {h_count}, 垂直图像数量: {v_count}")

for seq in number_dir:
    os.makedirs(os.path.join(target_dir,seq),exist_ok=True)
    os.makedirs(os.path.join(target_dir,seq,"rgb"),exist_ok=True)
    os.makedirs(os.path.join(target_dir,seq,"depth"),exist_ok=True)

    second_path = os.path.join(root+"/ca1m-train-"+seq,seq)
    frames_dir = os.listdir(second_path)
    frames_dir = [i.split(".")[0] for i in frames_dir if 'world' not in i]
    idx = sorted(np.unique(frames_dir))
    count = 0
    all_poses = []
    all_K_rgb = []
    all_K_depth = []
    T_gravity = []

    first_h = 0 
    first_w = 0

    gt_ins_json = os.path.join(second_path,'world.gt','instances.json')
    t_gt_ins_json = os.path.join(os.path.join(target_dir,seq),'instances.json')
    try:
        # 保留文件元数据（修改时间等）
        shutil.copy2(gt_ins_json, t_gt_ins_json)
        print(f"成功复制: {gt_ins_json} → {t_gt_ins_json}")
    except Exception as e:
        print(f"复制失败: {e}")

    for frame_id in idx:
        img_path = os.path.join(second_path,frame_id+'.wide')
        gt_path = os.path.join(second_path,frame_id+'.gt')

        rgb = os.path.join(img_path,'image.png')
        depth = os.path.join(gt_path,'depth.png')
        pose = os.path.join(gt_path,'RT.json')
        gravity = os.path.join(img_path,'T_gravity.json')
        K_rgb = os.path.join(gt_path,"image",'K.json')
        K_depth = os.path.join(gt_path,"depth",'K.json')


        cur_d  = cv2.imread(depth,-1)

        if first_h==0 and first_w==0:
            # 读取第一张图像的尺寸
            first_h, first_w = cur_d.shape[0], cur_d.shape[1]

        current_K_depth = None
        matrix = None
        with open(pose, 'r') as f:
            matrix = np.asarray(json.load(f))
            all_poses.append(matrix)
        with open(gravity, 'r') as f:
            matrix = np.asarray(json.load(f))
            T_gravity.append(matrix)


        if seq in complete_h_seqs or seq in complete_v_seqs:
            with open(K_rgb, 'r') as f:
                matrix = np.asarray(json.load(f))
                all_K_rgb.append(matrix)
            with open(K_depth, 'r') as f:
                current_K_depth = np.asarray(json.load(f))
                all_K_depth.append(current_K_depth)
        else:
            if seq in v_seqs:
                # 垂直序列，K_depth为0
                if cur_d.shape[0] > cur_d.shape[1]:
                    with open(K_rgb, 'r') as f:
                        matrix = np.asarray(json.load(f))
                        all_K_rgb.append(matrix)
                    with open(K_depth, 'r') as f:
                        current_K_depth = np.asarray(json.load(f))
                        all_K_depth.append(current_K_depth)
            else:
                if cur_d.shape[0] < cur_d.shape[1]:
                    with open(K_rgb, 'r') as f:
                        matrix = np.asarray(json.load(f))
                        all_K_rgb.append(matrix)
                    with open(K_depth, 'r') as f:
                        current_K_depth = np.asarray(json.load(f))
                        all_K_depth.append(current_K_depth)
        # else:
        #     # 如果当前图像的高度大于第一张图像的高度，则跳过该图像
        #     print(f"{seq} Skipping frame {frame_id} due to height mismatch. cur_d: {cur_d.shape[0]},{cur_d.shape[1]}, first_h: {first_h},{first_w}")
        # print(f"finishing {seq}")

        t_rgb = os.path.join(target_dir,seq,"rgb",str(count)+".png")
        t_depth = os.path.join(target_dir,seq,"depth",str(count)+".png")
        try:
            # 保留文件元数据（修改时间等）
            shutil.copy2(rgb, t_rgb)
            shutil.copy2(depth, t_depth)
            print(f"成功复制: {rgb} → {t_rgb}")
        except Exception as e:
            print(f"复制失败: {e}")
        count+=1
    # Save the pose and gravity data
    t_pose_path = os.path.join(target_dir,seq,"all_poses.npy")
    t_gravity_path = os.path.join(target_dir,seq,"T_gravity.npy")
    K_rgb_path = os.path.join(target_dir,seq,"K_rgb.txt")
    K_depth_path = os.path.join(target_dir,seq,"K_depth.txt")

    all_K_rgb = np.array(all_K_rgb)
    all_K_depth = np.array(all_K_depth)
    all_K_rgb = np.mean(all_K_rgb, axis=0)  # [3,3]
    all_K_depth = np.mean(all_K_depth, axis=0)  # [3,3]

    np.save(t_pose_path, np.array(all_poses)) #[N,4,4]
    np.save(t_gravity_path, np.array(T_gravity)) #[N,3,3]
    np.savetxt(K_rgb_path, np.array(all_K_rgb)) #[N,4,4]
    np.savetxt(K_depth_path, np.array(all_K_depth)) #[N,3,3]





