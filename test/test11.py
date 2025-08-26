import os
import re

def sort_folders_by_number(path):
    # 获取所有子文件夹名称
    folders = [f for f in os.listdir(path) 
               if os.path.isdir(os.path.join(path, f))]
    
    # 提取名称中的数字并排序
    def extract_number(name):
        numbers = re.findall(r'\d+', name)  # 提取连续数字
        return int(numbers[0]) if numbers else float('inf')  # 若无数字则置后
    
    sorted_folders = sorted(folders, key=extract_number)
    return sorted_folders

# 示例
path = "/media/lyq/gzr_1t/CA1M-dataset/test/"
sorted_folders = sort_folders_by_number(path)
# print(sorted_folders)
for folder in sorted_folders:
    print(folder)