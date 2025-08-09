import os
from PIL import Image
import uuid
import numpy as np


def analyze_image_orientation(folder_path):
    """
    分析文件夹中图片方向比例，并返回少数方向的图片文件名
    
    参数:
        folder_path: 图片文件夹路径
        
    返回:
        tuple: (主流方向描述, 少数方向图片列表)
    """
    # 初始化统计变量
    landscape_count = 0  # 横图计数器 (W > H)
    portrait_count = 0   # 竖图计数器 (H > W)
    square_count = 0     # 正方形计数器 (W == H)
    portrait_files = []  # 竖图文件名列表
    landscape_files = [] # 横图文件名列表
    
    # 支持的图片格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                    # 判断图片方向
                    if width > height:
                        landscape_count += 1
                        landscape_files.append(filename)
                    elif height > width:
                        portrait_count += 1
                        portrait_files.append(filename)
                    else:
                        square_count += 1
            except Exception as e:
                print(f"无法处理图片 {filename}: {e}")
    
    # 确定主流方向和需要输出的少数方向文件
    if landscape_count > portrait_count:
        print(f"横图数量: {landscape_count}, 竖图数量: {portrait_count}, 横图比例: {landscape_count/(landscape_count+portrait_count):.2f}, 竖图比例: {portrait_count/(landscape_count+portrait_count):.2f}")
        return ("横图为主(W>H)", portrait_files)
    elif portrait_count > landscape_count:
        print(f"横图数量: {landscape_count}, 竖图数量: {portrait_count}, 横图比例: {landscape_count/(landscape_count+portrait_count):.2f}, 竖图比例: {portrait_count/(landscape_count+portrait_count):.2f}")
        return ("竖图为主(H>W)", landscape_files)
    else:
        return ("横竖图数量相等", [])

def delete_minority_images(folder_path, minority_files):
    """删除不符合主流方向的图片"""
    deleted_count = 0
    for filename in minority_files:
        file_path = os.path.join(folder_path, filename)
        try:
            os.remove(file_path)
            print(f"已删除: {filename}")
            deleted_count += 1
        except Exception as e:
            print(f"删除 {filename} 失败: {e}")
    return deleted_count

def rename_remaining_images(folder_path):
    """
    对剩余图片按数字顺序重新命名，以.png前面的数字大小排序
    格式: 0.png, 1.jpg, ...
    """
    renamed_count = 0
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    
    # 获取所有图片文件
    try:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
        # 按文件名中的数字进行排序
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except (ValueError, FileNotFoundError) as e:
        print(f"无法读取或排序文件: {e}")
        return 0

    # 为了避免重命名冲突（例如将 2.png 重命名为 1.png，而 1.png 已存在），
    # 先将所有文件重命名为带唯一临时标记的名称。
    temp_files = []
    for old_name in files:
        old_path = os.path.join(folder_path, old_name)
        # 使用uuid确保临时文件名唯一
        temp_suffix = str(uuid.uuid4())
        file_ext = os.path.splitext(old_name)[1]
        temp_name = f"temp_{temp_suffix}{file_ext}"
        temp_path = os.path.join(folder_path, temp_name)
        try:
            os.rename(old_path, temp_path)
            temp_files.append((temp_path, file_ext))
        except Exception as e:
            print(f"无法将 {old_name} 重命名为临时文件: {e}")

    # 现在所有原始文件都已重命名，可以安全地将它们重命名为最终的序列名称
    for i, (temp_path, file_ext) in enumerate(temp_files):
        new_name = f"{i}{file_ext}"
        new_path = os.path.join(folder_path, new_name)
        try:
            os.rename(temp_path, new_path)
            renamed_count += 1
        except Exception as e:
            print(f"无法将临时文件重命名为 {new_name}: {e}")
            
    return renamed_count
    


if __name__ == "__main__":
    # 设置图片文件夹路径
    dataset_root = '/media/lyq/temp/dataset/train-CA-1M-slam'
    seq_list = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
    for scene_id in seq_list:
        root_path = f"{dataset_root}/{scene_id}/"  # 替换为你的实际路径
        rgb_folder = os.path.join(root_path, "rgb")
        depth_folder = os.path.join(root_path, "depth")
        # 分析图片方向
        main_orientation, minority_files = analyze_image_orientation(rgb_folder)
        
        if len(minority_files)==0:
            print(f"{scene_id} 图片方向一致，无需处理，跳过")
            continue
        
        # 输出分析结果
        print(f"方向分析结果: {main_orientation}")
        
        N_images = len(os.listdir(rgb_folder))
        delete_ids = [int(filename[:-4]) for filename in minority_files]

        valid_mask = np.ones(N_images, dtype=bool)
        valid_mask[delete_ids] = False

        # 步骤2: 删除all_poses.npy中对应的位姿
        all_poses_path = os.path.join(root_path, "all_poses.npy")
        if os.path.exists(all_poses_path):
            try:
                all_poses = np.load(all_poses_path)
                if all_poses.shape[0] == N_images:
                    # print(f"原始 all_poses.npy 形状: {all_poses.shape}")
                    filtered_poses = all_poses[valid_mask]
                    np.save(all_poses_path, filtered_poses)
                    # print(f"已更新 all_poses.npy, 新形状: {filtered_poses.shape}")
                else:
                    print(f"警告: all_poses.npy 中的位姿数量 ({all_poses.shape[0]}) 与图片数量 ({N_images}) 不匹配。")
            except Exception as e:
                print(f"处理 all_poses.npy 时出错: {e}")
        else:
            print(f"未找到 all_poses.npy 文件: {all_poses_path}")
        
        # 步骤3: 删除不符合主流方向的图片
        if len(minority_files)>0:
            if N_images > 0 and (len(minority_files) / N_images) < 0.3:
                print(f"\n发现 {len(minority_files)} 张少数方向图片，比例低于30%，将自动删除。")
                deleted_count = delete_minority_images(rgb_folder, minority_files)
                print(f"\n已删除 {deleted_count} 张图片")
            else:
                print(f"\n发现 {len(minority_files)} 张少数方向图片，但比例超过30%，跳过删除操作。")
        else:
            print("没有需要删除的图片")
        
        
        # 步骤4: 对剩余图片重新命名
        print("\n正在对剩余图片重新命名...")
        renamed_count = rename_remaining_images(rgb_folder)
        print(f"\n已完成 {renamed_count} 张图片的重命名")
        print("所有操作已完成！")
        
        # 使用星号边框 + 动态结束符
        print("\n" + "*" * 30)
        print(f"***  \033[1;36mRGB图像处理完成！\033[0m  ***")  # 36=青色，1=粗体
        print("*" * 30 + "🎉\n")
        
        # 步骤5: 删除不符合主流方向的图片
        if len(minority_files)>0:
            if N_images > 0 and (len(minority_files) / N_images) < 0.3:
                print(f"\n发现 {len(minority_files)} 张少数方向图片，比例低于30%，将自动删除。")
                deleted_count = delete_minority_images(depth_folder, minority_files)
                print(f"\n已删除 {deleted_count} 张图片")
            else:
                print(f"\n发现 {len(minority_files)} 张少数方向图片，但比例超过30%，跳过删除操作。")
        else:
            print("没有需要删除的图片")
        
        # 步骤6: 对剩余图片重新命名
        print("\n正在对剩余图片重新命名...")
        renamed_count = rename_remaining_images(depth_folder)
        print(f"\n已完成 {renamed_count} 张图片的重命名")
        print("所有操作已完成！")
        
        print("\n" + "*" * 30)
        print(f"***  \033[1;36mDepth图像处理完成！\033[0m  ***")  # 36=青色，1=粗体
        print("*" * 30 + "🎉\n")
        
