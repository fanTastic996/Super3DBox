import json
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
import open3d as o3d


def random_color_v2(value, maximum=255):
    """生成基于归一化值的颜色映射"""
    jet_cmap = get_cmap('jet')
    rgba = jet_cmap(value)
    return np.array(rgba[:3])  # 返回 RGB 分量

def visualize_bboxes_from_json(json_path, mesh_path):
    

    """从 JSON 文件加载并可视化 3D 边界框"""
    # 初始化 Rerun
    rr.init("3d_bbox_visualization", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)  # 设定世界坐标系[7](@ref)timeless=True)  # Y轴向上坐标系
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)  
    vertex_colors = np.asarray(mesh.vertex_colors)
    
    rr.log(
            "world/points", 
            rr.Points3D(vertices, colors=vertex_colors, radii=0.005)  # 控制点云半径
    )
    
    
    
    # 1. 加载 JSON 数据
    all_centers = []
    all_sizes = []
    all_rot = []
    with open(json_path, 'r') as f:
        #['id', 'category', 'position', 'scale', 'R', 'corners']
        data = json.load(f)
        for i in range(len(data)):
            center = np.array(data[i]['position'])
            size = np.array(data[i]['scale'])
            # size = np.array([data[i]['scale'][1], data[i]['scale'][0], data[i]['scale'][2]])
            
            rotation_matrix = np.array(data[i]['R'])
            all_centers.append(center)
            all_sizes.append(size)
            all_rot.append(rotation_matrix)
    all_centers = np.array(all_centers)
    all_sizes = np.array(all_sizes)
    # print("all_sizes", all_sizes)
    all_rot = np.array(all_rot)
    
    # 2. 生成 3D  colors
    all_colors = [random_color_v2(ind/(all_centers.shape[0])) for ind in range(all_centers.shape[0])]
    
  

    quaternions = [
        rr.Quaternion(
            xyzw=Rotation.from_matrix(r).as_quat()
        )
        for r in all_rot
    ]

    
    # 7. 生成标签 ID
    labels = [f"obj_{i}" for i in range(all_centers.shape[0])]
    
    # 8. 使用 Rerun 可视化 [9](@ref)
    rr.log(
        "world/bboxes",
        rr.Boxes3D(
            centers=all_centers,
            sizes=all_sizes,
            quaternions=quaternions,
            colors=all_colors,
            labels=labels
        )
    )
    print(f"成功可视化 {all_centers.shape[0]} 个 3D 边界框")

if __name__ == "__main__":
    # 使用示例
    visualize_bboxes_from_json("/media/lyq/temp/dataset/train-CA-1M-slam/42897951/instances.json", "/media/lyq/temp/dataset/train-CA-1M-slam/42897951/mesh.ply")
    
