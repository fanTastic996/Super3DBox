import os
import json
import numpy as np

def process_json_file(file_path):
    """
    处理单个 JSON 文件：删除 'corners' 值为空列表的字典项
    """
    try:
        # 读取 JSON 文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 过滤掉 'corners' 为空的项
        filtered_data = [item for item in data if item.get('corners') != []]
        
        # 保存修改后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)
        
        # 返回处理统计信息
        return len(data), len(filtered_data)
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def process_all_folders(root_dir):
    """
    遍历根目录下所有子文件夹，处理其中的 _transformed.json 文件
    """
    total_files = 0
    total_removed = 0
    train_list = np.loadtxt(os.path.join(root_dir, 'train_list.txt'), dtype=str)
    # 遍历所有子文件夹
    for seq in train_list:

        # 检查是否存在目标 JSON 文件
        
        file_path = os.path.join(root_dir, seq , seq+ '_sampling_results_transformed.json')
        print(f"处理中: {file_path}")
        
        # 处理文件并获取结果
        result = process_json_file(file_path)
        
        if result:
            original_count, filtered_count = result
            removed = original_count - filtered_count
            total_removed += removed
            total_files += 1
            print(f"  - 原始条目: {original_count}, 保留条目: {filtered_count}, 删除条目: {removed}")
    
    # 输出最终统计
    print("\n处理完成!")
    print(f"总共处理文件: {total_files}")
    print(f"总共删除空项: {total_removed}")

if __name__ == "__main__":
    # 设置要处理的根目录路径
    root_directory = "/media/lyq/temp/dataset/train-CA-1M-slam/"  # 替换为实际路径
    
    # 执行处理
    process_all_folders(root_directory)