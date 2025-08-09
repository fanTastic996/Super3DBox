import os.path as osp
import numpy as np
import json

instance_json_path = '/media/lyq/temp/dataset/train-CA-1M-slam/42444499/instances.json'
with open(instance_json_path, 'r') as f:
    instances_data = json.load(f)

# instances_data: dict_keys(['id', 'category', 'position', 'scale', 'R', 'corners']) 84
print("instances_data:", np.array(instances_data[0]['corners']).shape, len(instances_data))