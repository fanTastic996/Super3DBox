import os
import argparse
import numpy as np
def find_results_folders(root_dir):
    """
    递归查找包含 _results.json 文件的子文件夹
    
    参数:
        root_dir (str): 要搜索的根目录路径
        
    返回:
        list: 包含 _results.json 文件的子文件夹路径列表
    """
    result_folders = []
    
    # 遍历根目录下的所有目录（使用 os.walk 实现递归遍历）
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 检查当前目录中是否存在 _results.json 文件
        # print(filenames)
        for i in filenames:
            if '_results_transformed.json' in i:
                result_folders.append(dirpath)
    
    return result_folders

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='查找包含 _results.json 文件的子文件夹')
    parser.add_argument('directory', type=str, help='要搜索的根目录路径')
    args = parser.parse_args()
    
    # 验证目录是否存在
    if not os.path.isdir(args.directory):
        print(f"错误: 目录 '{args.directory}' 不存在")
        exit(1)
    
    # 执行搜索
    found_folders = find_results_folders(args.directory)
    # print("found_folders:", found_folders)
    train_list = [i.split('/')[-1] for i in found_folders]
    print("train_list:", train_list)
    # 输出结果
    if found_folders:
        print(f"找到 {len(found_folders)} 个包含 _results.json 的文件夹:")
        # for folder in found_folders:
        #     print(f"  → {folder}")
    else:
        print("未找到包含 _results.json 文件的子文件夹")
        
    np.savetxt(os.path.join(args.directory,'train_list.txt'), train_list, fmt='%s', delimiter='\n')




#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# import json
# test_json = '/media/lyq/temp/dataset/train-CA-1M-slam/47204724/47204724_sampling_results.json'
# with open(test_json, 'r') as f:
#     seq_data = json.load(f)
# print("len seq_data:", seq_data[0].keys())